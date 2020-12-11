namespace Microsoft.Build.CPPTasks
{
    using System;
    using System.Collections.Generic;
    using System.Text;
    using System.Text.RegularExpressions;
    using System.IO;
    using System.Runtime.InteropServices;
    using System.Threading;
    using System.Collections.Specialized;
    using System.Resources;
    using Microsoft.Build.Tasks;
    using Microsoft.Build.Framework;
    using Microsoft.Build.Utilities;
    using Microsoft.Build.Shared;
    using Microsoft.Win32.SafeHandles;
    using System.Linq;
    using System.Diagnostics;

    /// <summary>
    /// A data-driven task that uses FileTracker incremental build
    /// Handles all the FileTracker work for a straightforward task (like RC),
    /// and most of the work for a complex one (like CL)
    /// </summary>
    public abstract class TrackedVCToolTask : VCToolTask
    {
#if WHIDBEY_VISIBILITY
        internal
#else
        protected
#endif
        TrackedVCToolTask(ResourceManager taskResources)
            : base(taskResources)
        {
            PostBuildTrackingCleanup = true;
            EnableExecuteTool = true;
        }

        /// <summary>
        /// Did this task skip its execution because it was up to date
        /// </summary>
        private bool skippedExecution = false;

        /// <summary>
        /// Input dependency graph
        /// </summary>
        private CanonicalTrackedInputFiles sourceDependencies;
        private CanonicalTrackedOutputFiles sourceOutputs;

        /// <summary>
        /// Whether to track file reads and writes or not
        /// </summary>
        private bool trackFileAccess = false;

        /// <summary>
        /// Whether to track command line between builds
        /// </summary>
        private bool trackCommandLines = true;

        /// <summary>
        /// Whether to do an incremental build
        /// </summary>
        private bool minimalRebuildFromTracking = false;

        /// <summary>
        /// Flag to delete the Outputs when the tool is going to start (default false)
        /// </summary>
        private bool deleteOutputBeforeExecute = false;

        /// <summary>
        /// root sources
        /// </summary>
        private string rootSource;

        /// <summary>
        /// read files
        /// </summary>
        private ITaskItem[] tlogReadFiles = null;

        /// <summary>
        /// write files
        /// </summary>
        private ITaskItem[] tlogWriteFiles = null;

        /// <summary>
        /// Command tlog
        /// </summary>
        private ITaskItem tlogCommandFile = null;

        /// <summary>
        /// Source that was actually compiled, or will be.
        /// </summary>
        private ITaskItem[] sourcesCompiled;

        /// <summary>
        /// Files that should be compacted out of the input tlogs.
        /// </summary>
        private ITaskItem[] trackedInputFilesToIgnore = null;

        /// <summary>
        /// Files that should be compacted out of the output tlogs.
        /// </summary>
        private ITaskItem[] trackedOutputFilesToIgnore = null;

        /// <summary>
        /// default excluded directory from input tracking
        /// </summary>
        /// <remarks>
        /// Ignoring example C:\Windows, C:\Windows\System, C:\Windows\System32, C:\WINDOWS\GLOBALIZATION\SORTING
        /// These directories contains files used by cl and linker that may changed during Windows Update.
        /// </remarks>
        private static TaskItem[] excludedInputPathsDefault = new TaskItem[] {
            new TaskItem(Environment.GetFolderPath(Environment.SpecialFolder.Windows)),
            new TaskItem(Environment.GetFolderPath(Environment.SpecialFolder.System)),
            new TaskItem(Environment.GetFolderPath(Environment.SpecialFolder.SystemX86)),
            new TaskItem(Environment.GetFolderPath(Environment.SpecialFolder.Windows)+@"\GLOBALIZATION\SORTING"),
        };

        /// <summary>
        /// Paths from which dependencies will be ignored during up to date checking
        /// </summary>
        private ITaskItem[] excludedInputPaths = excludedInputPathsDefault;

        /// <summary>
        /// Intermediate directory for tracker to write tlogs into.
        /// Individual tasks will implement this to pass in a property like TrackerLogDirectory
        /// </summary>
        protected abstract string TrackerIntermediateDirectory
        {
            get;
        }

        /// <summary>
        /// Files that should be considered the root sources for this tool.
        /// Individual tasks will implement this to pass in a property like Source or Sources.
        /// </summary>
        protected abstract ITaskItem[] TrackedInputFiles
        {
            get;
        }

        /// <summary>
        /// Check an environment variable to see if we want to use unicode pipes for communicating
        /// with cl.exe and link.exe.
        /// </summary>
        protected bool VCToolTaskUseUnicodeOutput
        {
            get
            {
                var useUnicodeOutput = Environment.GetEnvironmentVariable("VCTOOLTASK_USE_UNICODE_OUTPUT");
                return useUnicodeOutput != null && bool.TryParse(useUnicodeOutput, out bool useUnicodeOutputValue) && useUnicodeOutputValue;
            }
        }

        /// <summary>
        /// Input dependency graph
        /// </summary>
#if WHIDBEY_VISIBILITY
        internal
#else
        protected
#endif
        CanonicalTrackedInputFiles SourceDependencies
        {
            get
            {
                return sourceDependencies;
            }
            set
            {
                sourceDependencies = value;
            }
        }

        /// <summary>
        /// Input dependency graph
        /// </summary>
#if WHIDBEY_VISIBILITY
        internal
#else
        protected
#endif
        CanonicalTrackedOutputFiles SourceOutputs
        {
            get
            {
                return sourceOutputs;
            }
            set
            {
                sourceOutputs = value;
            }
        }

        /// <summary>
        /// Did this task skip its execution because it was up to date
        /// </summary>
        [Output]
        public bool SkippedExecution
        {
            get
            {
                return skippedExecution;
            }
            set
            {
                this.skippedExecution = value;
            }
        }

        /// <summary>
        /// Root marker. Normally this is just based on the sources compiled,
        /// but some tasks may want to override it. For example, CL may incorporate
        /// the output preprocessed file into its root marker because a single input can
        /// preprocess to several different outputs.
        /// </summary>
        public string RootSource
        {
            get
            {
                return rootSource;
            }
            set
            {
                rootSource = value;
            }
        }

        /// <summary>
        /// Whether to track ReplaceFile API. Derived classes can override value.
        /// </summary>
        protected virtual bool TrackReplaceFile
        {
            get
            {
                return false;
            }
        }

        /// <summary>
        /// Names of the read tlogs for this tool
        /// </summary>
        protected virtual string[] ReadTLogNames
        {
            get
            {
                string toolName = Path.GetFileNameWithoutExtension(ToolExe);
                return new string[] { toolName + ".read.*.tlog", toolName + ".*.read.*.tlog", toolName + "-*.read.*.tlog", this.GetType().FullName + ".read.*.tlog" };
            }
        }

        /// <summary>
        /// Names of the write tlogs for this tool
        /// </summary>
        protected virtual string[] WriteTLogNames
        {
            get
            {
                string toolName = Path.GetFileNameWithoutExtension(ToolExe);
                return new string[] { toolName + ".write.*.tlog", toolName + ".*.write.*.tlog", toolName + "-*.write.*.tlog", this.GetType().FullName + ".write.*.tlog" };
            }
        }

        /// <summary>
        /// Names of the write tlogs for this tool
        /// </summary>
        protected virtual string[] DeleteTLogNames
        {
            get
            {
                string toolName = Path.GetFileNameWithoutExtension(ToolExe);
                return new string[] { toolName + ".delete.*.tlog", toolName + ".*.delete.*.tlog", toolName + "-*.delete.*.tlog", this.GetType().FullName + ".delete.*.tlog" };
            }
        }

        /// <summary>
        /// Name of the commandline tlog for this tool
        /// </summary>
        protected virtual string CommandTLogName
        {
            get
            {
                string toolName = Path.GetFileNameWithoutExtension(ToolExe);
                return toolName + ".command.1.tlog";
            }
        }

        /// <summary>
        /// TLog read files that the filetracker writes during tool execution (optional)
        /// </summary>
        /// <owner>KieranMo</owner>
        /// <returns>ITaskItem array of tlog files</returns>
        public ITaskItem[] TLogReadFiles
        {
            get
            {
                return this.tlogReadFiles;
            }
            set
            {
                this.tlogReadFiles = value;
            }
        }

        /// <summary>
        /// TLog write files that the filetracker writes during tool execution (optional)
        /// </summary>
        /// <owner>KieranMo</owner>
        /// <returns>ITaskItem array of tlog files</returns>
        public ITaskItem[] TLogWriteFiles
        {
            get
            {
                return this.tlogWriteFiles;
            }
            set
            {
                this.tlogWriteFiles = value;
            }
        }

        /// <summary>
        /// TLog command file that the task writes during tool execution.
        /// </summary>
        public ITaskItem TLogCommandFile
        {
            get
            {
                return this.tlogCommandFile;
            }
            set
            {
                this.tlogCommandFile = value;
            }
        }

        /// <summary>
        /// Whether to track file accesses for this task
        /// Typically this is enabled for both clean and incremental builds
        /// </summary>
        public bool TrackFileAccess
        {
            get
            {
                return trackFileAccess;
            }
            set
            {
                trackFileAccess = value;
            }
        }

        /// <summary>
        /// Whether to track command lines for this task
        /// Typically this is enabled.  When disabled, command lines would still be writen to disk
        /// to allow it continue to work when re-enable again.
        /// </summary>
        public bool TrackCommandLines
        {
            get
            {
                return trackCommandLines;
            }
            set
            {
                trackCommandLines = value;
            }
        }

        /// <summary>
        /// Enable or Disable the tlog clean up after the build.  Default enabled.
        /// </summary>
        public bool PostBuildTrackingCleanup
        {
            get;
            set;
        }

        /// <summary>
        /// Enable or Disable running the tool.  When disabled, PostBuildTrackingCleanup is also disabled.
        /// </summary>
        public bool EnableExecuteTool
        {
            get;
            set;
        }

        /// <summary>
        /// Minimally rebuild the sources based on tracked accesses
        /// In other words, whether to do an incremental build
        /// </summary>
        public bool MinimalRebuildFromTracking
        {
            get
            {
                return minimalRebuildFromTracking;
            }
            set
            {
                minimalRebuildFromTracking = value;
            }
        }

        /// <summary>
        /// Enable extended tracking: GetFileAttributes, GetFileAttributesEx
        /// RemoveDirectory, CreateDirectory
        /// </summary>
        public virtual bool AttributeFileTracking
        {
            get
            {
                return false;
            }
        }

        /// <summary>
        /// The compiled sources (or the subset that we're
        /// about to compile, depending on where we are)
        /// </summary>
        [Output]
        public ITaskItem[] SourcesCompiled
        {
            get
            {
                return sourcesCompiled;
            }
            set
            {
                sourcesCompiled = value;
            }
        }

        /// <summary>
        /// Files that should be compacted out of the output tlogs.
        /// </summary>
        public ITaskItem[] TrackedOutputFilesToIgnore
        {
            get { return trackedOutputFilesToIgnore; }
            set { trackedOutputFilesToIgnore = value; }
        }

        /// <summary>
        /// Files that should be compacted out of the input tlogs.
        /// </summary>
        public ITaskItem[] TrackedInputFilesToIgnore
        {
            get { return trackedInputFilesToIgnore; }
            set { trackedInputFilesToIgnore = value; }
        }

        /// <summary>
        /// Flag to delete the Outputs when the tool is going to start (default false)
        /// </summary>
        /// <remarks>Renamed to DeleteOutputBeforeExecute</remarks>
        public bool DeleteOutputOnExecute
        {
            get { return deleteOutputBeforeExecute; }
            set { deleteOutputBeforeExecute = value; }
        }

        /// <summary>
        /// Flag to delete the Outputs when the tool is going to start (default false)
        /// </summary>
        /// <remarks>Renamed from DeleteOutputOnExecute</remarks>
        public bool DeleteOutputBeforeExecute
        {
            get { return deleteOutputBeforeExecute; }
            set { deleteOutputBeforeExecute = value; }
        }

        /// <summary>
        /// True if composite rooting markers should not be compacted out
        /// of the tracking logs.  Default behavior is to compact.
        /// </summary>
        protected virtual bool MaintainCompositeRootingMarkers
        {
            get { return false; }
        }

        /// <summary>
        /// True means that the command line will only contain the those
        /// files out of date. False if all source files need to be passed
        /// on the command line when at least one of them is out of date.
        /// WARNING: Minimal rebuild optimization requires 100% accurate computed
        /// outputs to be specified!
        /// </summary>
        protected virtual bool UseMinimalRebuildOptimization
        {
            get { return false; }
        }

        /// <summary>
        /// Indicates the name of the property that contains the "source files"
        /// of this task.  Defaults to "Sources".
        /// </summary>
        public virtual string SourcesPropertyName
        {
            get { return "Sources"; }
        }

        /// <summary>
        /// Returns the ToolType of the wrapped tool.  This is null by
        /// default and should only be non-null if the bitness of the tool
        /// is being determined by something other than its location on
        /// the path.
        /// </summary>
        protected virtual ExecutableType? ToolType
        {
            get { return null; }
        }

        /// <summary>
        /// Returns the architecture of the tool being executed.  Should be a
        /// member of the Microsoft.Build.Utilities.ExecutableType enum.
        /// </summary>
        public string ToolArchitecture
        {
            get;
            set;
        }

        /// <summary>
        /// Path to the appropriate .NET Framework location that contains FileTracker.dll.  If set, the user
        /// takes responsibility for making sure that the bitness of the FileTracker.dll that they pass matches
        /// the bitness of the tool that they intend to use.
        /// </summary>
        /// <comments>
        /// Should only need to be used in partial or full checked in toolset scenarios.
        /// </comments>
        public string TrackerFrameworkPath
        {
            get;
            set;
        }

        /// <summary>
        /// Path to the appropriate Windows SDK location that contains Tracker.exe.  If set, the user takes
        /// responsibility for making sure that the bitness of the Tracker.exe that they pass matches the
        /// bitness of the tool that they intend to use.
        /// </summary>
        /// <comments>
        /// Should only need to be used in partial or full checked in toolset scenarios.
        /// </comments>
        public string TrackerSdkPath
        {
            get;
            set;
        }

        /// <summary>
        /// That set of paths from which tracked inputs will be ignored during
        /// Up to date checking
        /// </summary>
        public ITaskItem[] ExcludedInputPaths
        {
            get
            {
                return excludedInputPaths;
            }
            set
            {
                // Append to the default excluded set
                List<ITaskItem> items = new List<ITaskItem>(value);
                items.AddRange(excludedInputPathsDefault);
                excludedInputPaths = items.ToArray();
            }
        }

        /// <summary>
        /// Assign the default TLog Filenames for this task
        /// The names are gathered from ReadTLogNames and WriteTLogNames properties
        /// </summary>
        protected virtual void AssignDefaultTLogPaths()
        {
            // Only if the tlog files haven't been specified should we provide them here
            var trackerIntermediateDirectory = TrackerIntermediateDirectory;
            if (TLogReadFiles == null)
            {
                var tlogNames = ReadTLogNames;
                TLogReadFiles = new ITaskItem[tlogNames.Length];
                for (int i = 0; i < tlogNames.Length; i++)
                {
                    TLogReadFiles[i] = new TaskItem(Path.Combine(trackerIntermediateDirectory, tlogNames[i]));
                }
            }

            if (TLogWriteFiles == null)
            {
                var tlogNames = WriteTLogNames;
                TLogWriteFiles = new ITaskItem[tlogNames.Length];
                for (int i = 0; i < tlogNames.Length; i++)
                {
                    TLogWriteFiles[i] = new TaskItem(Path.Combine(trackerIntermediateDirectory, tlogNames[i]));
                }
            }

            if (TLogCommandFile == null)
            {
                TLogCommandFile = new TaskItem(Path.Combine(trackerIntermediateDirectory, CommandTLogName));
            }
        }

        /// <summary>
        /// Returns true if task execution is not necessary.
        /// </summary>
        /// <returns></returns>
        /// <owner>danmose/kieranmo</owner>
        override protected bool SkipTaskExecution()
        {
            return ComputeOutOfDateSources();
        }

        /// <summary>
        /// Returns true if all sources are up-to-date
        /// Updates "SourcesCompiled" property with the set of out-of-date sources.
        /// </summary>
        protected internal virtual bool ComputeOutOfDateSources()
        {
            if (MinimalRebuildFromTracking || TrackFileAccess)
            {
                // Set up the paths to the read and write tlogs
                AssignDefaultTLogPaths();
            }

            if (MinimalRebuildFromTracking && !ForcedRebuildRequired())
            {
                // Read the output graph
                sourceOutputs = new CanonicalTrackedOutputFiles(this, TLogWriteFiles);

                // Figure out the dependencies
                sourceDependencies = new CanonicalTrackedInputFiles(this, TLogReadFiles, TrackedInputFiles, ExcludedInputPaths, sourceOutputs, UseMinimalRebuildOptimization, MaintainCompositeRootingMarkers);

                // Figure out what is out of date
                ITaskItem[] sourcesOutOfDateThroughTracking = SourceDependencies.ComputeSourcesNeedingCompilation(false /* we don't want to look into rooting markers */);
                List<ITaskItem> sourcesWithChangedCommandLines = GenerateSourcesOutOfDateDueToCommandLine();

                SourcesCompiled = MergeOutOfDateSourceLists(sourcesOutOfDateThroughTracking, sourcesWithChangedCommandLines);

                if (SourcesCompiled.Length == 0)
                {
                    SkippedExecution = true;
                    // That was easy.
                    return SkippedExecution;
                }
                else
                {
                    // Restrict command line to the out of date subset of sources
                    SourcesCompiled = AssignOutOfDateSources(SourcesCompiled);

                    // Before executing the tool, remove information for these sources,
                    // because the tool is about to cause updated information to be logged.
                    // Remove from graph
                    SourceDependencies.RemoveEntriesForSource(SourcesCompiled);
                    SourceDependencies.SaveTlog();

                    if (DeleteOutputOnExecute)
                    {
                        // OutputsForSource() has a bug that will find other outputs with similar name.
                        // ie, Foo.c will also return Foo.c.Bar.c.
                        // Setting second arguement to searchForSubRootsInCompositeRootingMarkers=false to workaround it.
                        DeleteFiles(sourceOutputs.OutputsForSource(SourcesCompiled, false));
                    }

                    // Remove from outputs tlog
                    // QUESTION: why not remove from inputs tlog?
                    sourceOutputs.RemoveEntriesForSource(SourcesCompiled);
                    sourceOutputs.SaveTlog();
                }
            }
            else
            {
                // Not doing minimal rebuild: use all sources
                SourcesCompiled = TrackedInputFiles;

                if (SourcesCompiled == null || SourcesCompiled.Length == 0)
                {
                    SkippedExecution = true;
                    // That was easy.
                    return SkippedExecution;
                }
            }

            if ((TrackFileAccess || TrackCommandLines) && String.IsNullOrEmpty(RootSource))
            {
                // Store the root marker
                RootSource = FileTracker.FormatRootingMarker(SourcesCompiled);
            }

            SkippedExecution = false;
            // That was easy.
            return SkippedExecution;
        }

        /// <summary>
        /// Called to modify the command line if necessary to restrict it to
        /// the particular out of date subset of sources.
        /// </summary>
        protected virtual ITaskItem[] AssignOutOfDateSources(ITaskItem[] sources)
        {
            // General case is to do nothing, because most tools either run
            // on all sources or no sources; so we never need to modify the
            // command line here to be a subset of sources.
            // CL overrides this because it can actually correlate inputs and outputs.
            // CustomBuild overrides this as well. It may add more items to the list, based on its dependency analysis
            return sources;
        }

        /// <summary>
        /// Returns true if there is some condition set by the task that's not covered by the
        /// tlogs, which would require a full build to be done instead of incremental.
        /// E.g. CL will override this because the tlogs don't know about the PDB, so it
        /// needs to force a rebuild if the PDB is expected to exist but is not there.
        /// By default, it checks to make sure that the command tlog (required for comparing
        /// command lines) is there.
        /// </summary>
        /// <returns></returns>
        protected virtual bool ForcedRebuildRequired()
        {
            string commandLogPath = null;

            try
            {
                commandLogPath = TLogCommandFile.GetMetadata("FullPath");
            }
            catch (Exception e)
            {
                if (!(e is InvalidOperationException) &&
                    !(e is NullReferenceException))
                {
                    throw;
                }

                // We want to catch and log the error if the tlog name is bad
                Log.LogWarningWithCodeFromResources("TrackedVCToolTask.RebuildingDueToInvalidTLog", e.Message);
                return true;
            }

            // Otherwise, it is at least a valid path.
            if (!File.Exists(commandLogPath))
            {
                Log.LogMessageFromResources(MessageImportance.Low, "TrackedVCToolTask.RebuildingNoCommandTLog", TLogCommandFile.GetMetadata("FullPath"));
                return true;
            }

            return false;
        }

        /// <summary>
        /// Generates the list of sources that should be rebuilt because their commandline has changed
        /// since the last time the project was built.
        /// </summary>
        /// <returns></returns>
        protected virtual List<ITaskItem> GenerateSourcesOutOfDateDueToCommandLine()
        {
            // Generate sources-to-commandlines hash
            IDictionary<string, string> sourcesToCommandLines = MapSourcesToCommandLines();
            List<ITaskItem> sourcesStillOutOfDate = new List<ITaskItem>();

            // if not tracking command lines, then return empty so all command lines matches.
            if (!TrackCommandLines)
            {
                return sourcesStillOutOfDate;
            }

            if (sourcesToCommandLines.Count == 0)
            {
                // There was no log file, or some sort of catastrophic failure.  This should never happen (it ought
                // to be caught before we reach this point), but in case it does, short-circuit all the calculations.

                foreach (ITaskItem source in TrackedInputFiles)
                {
                    sourcesStillOutOfDate.Add(source);
                }
            }
            else
            {
                if (MaintainCompositeRootingMarkers)
                {
                    // If we're maintaining the rooting markers, our command line is the full command line.
                    string newCommandLine = ApplyPrecompareCommandFilter(GenerateCommandLine(CommandLineFormat.ForTracking));

                    string priorCommandLine = null;
                    if (sourcesToCommandLines.TryGetValue(FileTracker.FormatRootingMarker(TrackedInputFiles), out priorCommandLine))
                    {
                        // we have a command line.  If it matches, everything is up-to-date (at least as far as
                        // the command line is concerned).  Otherwise, everything is out-of-date, since we only
                        // have the one shared command line.
                        // Must do case-sensitive compare since there are some switches for which case does matter
                        priorCommandLine = ApplyPrecompareCommandFilter(priorCommandLine);
                        if (priorCommandLine == null || !newCommandLine.Equals(priorCommandLine, StringComparison.Ordinal))
                        {
                            foreach (ITaskItem source in TrackedInputFiles)
                            {
                                sourcesStillOutOfDate.Add(source);
                            }
                        }
                    }
                    else
                    {
                        foreach (ITaskItem source in TrackedInputFiles)
                        {
                            sourcesStillOutOfDate.Add(source);
                        }
                    }
                }
                else
                {
                    // Otherwise, we have a separate command line for each source, so as not to break
                    // minimal rebuild optimization.  So just generate the source-agnostic part of the
                    // command line here.
                    string sourcesPropertyName = SourcesPropertyName ?? "Sources";
                    string newCommandLineBase = GenerateCommandLineExceptSwitches(new string[] { sourcesPropertyName }, CommandLineFormat.ForTracking);

                    foreach (ITaskItem source in TrackedInputFiles)
                    {
                        string newCommandLine = ApplyPrecompareCommandFilter(newCommandLineBase + " " + source.GetMetadata("FullPath").ToUpperInvariant());
                        string priorCommandLine = null;

                        if (sourcesToCommandLines.TryGetValue(FileTracker.FormatRootingMarker(source), out priorCommandLine))
                        {
                            // we have a command line.  If they match, everything is good.  Otherwise, this source is
                            // out-of-date and should be treated accordingly.
                            // Must do case-sensitive compare since there are some switches for which case does matter
                            priorCommandLine = ApplyPrecompareCommandFilter(priorCommandLine);
                            if (priorCommandLine == null || !newCommandLine.Equals(priorCommandLine, StringComparison.Ordinal))
                            {
                                sourcesStillOutOfDate.Add(source);
                            }
                        }
                        else
                        {
                            sourcesStillOutOfDate.Add(source);
                        }
                    }
                }
            }

            return sourcesStillOutOfDate;
        }

        /// <summary>
        /// Given the lists of out-of-date sources, merge them into a single list that preserves the
        /// order of the original sources.
        /// </summary>
        /// <param name="sourcesOutOfDateThroughTracking">Sources that have been marked as out-of-date
        /// because of information discovered from the tracking logs.</param>
        /// <param name="sourcesWithChangedCommandLines">Sources that have been marked as out-of-date
        /// because their command lines have changed since the last time they built.</param>
        /// <returns></returns>
        protected ITaskItem[] MergeOutOfDateSourceLists(ITaskItem[] sourcesOutOfDateThroughTracking, List<ITaskItem> sourcesWithChangedCommandLines)
        {
            // Simple checks first:

            // Are there no out-of-date sources because of command lines?  Then it's just the tracked ones
            // that count.
            if (sourcesWithChangedCommandLines.Count == 0)
            {
                return sourcesOutOfDateThroughTracking;
            }

            // Are there no out-of-date sources because of tracking?  Then it's just the ones with command lines
            if (sourcesOutOfDateThroughTracking.Length == 0)
            {
                if (sourcesWithChangedCommandLines.Count == TrackedInputFiles.Length)
                {
                    Log.LogMessageFromResources(MessageImportance.Low, "TrackedVCToolTask.RebuildingAllSourcesCommandLineChanged");
                }
                else
                {
                    foreach (ITaskItem source in sourcesWithChangedCommandLines)
                    {
                        Log.LogMessageFromResources(MessageImportance.Low, "TrackedVCToolTask.RebuildingSourceCommandLineChanged", source.GetMetadata("FullPath"));
                    }
                }
                return sourcesWithChangedCommandLines.ToArray();
            }

            // Are all the files out-of-date?
            if (sourcesOutOfDateThroughTracking.Length == TrackedInputFiles.Length)
            {
                return TrackedInputFiles;
            }

            // ... in either case?
            if (sourcesWithChangedCommandLines.Count == TrackedInputFiles.Length)
            {
                Log.LogMessageFromResources(MessageImportance.Low, "TrackedVCToolTask.RebuildingAllSourcesCommandLineChanged");

                return TrackedInputFiles;
            }

            // Simple checks have all failed -- now we actually have to merge.

            // sourcesOutOfDateThroughTracking is sorted in *case-sensitive* order, but TrackedInputFiles
            // is in the same order as it was passed to us -- the same order as in the project file --
            // and sourcesWithChangedCommandLines is a subset of TrackedInputFiles ordered in the same way.

            // We need to make sure that we compile any out-of-date source files in the same order as in the
            // project file -- e.g., the same order as TrackedInputFiles.

            Dictionary<ITaskItem, bool> outOfDateSources = new Dictionary<ITaskItem, bool>();

            // use the bool to indicate whether the command line was changed or not -- if it was changed, we will want to
            // output a message.
            foreach (ITaskItem trackedSource in sourcesOutOfDateThroughTracking)
            {
                outOfDateSources[trackedSource] = false; // not derived from command line changing
            }

            foreach (ITaskItem source in sourcesWithChangedCommandLines)
            {
                if (!outOfDateSources.ContainsKey(source))
                {
                    outOfDateSources.Add(source, true /* command line */);
                }
            }

            List<ITaskItem> mergedSources = new List<ITaskItem>();
            foreach (ITaskItem source in TrackedInputFiles)
            {
                bool fromCommandLine = false;
                if (outOfDateSources.TryGetValue(source, out fromCommandLine))
                {
                    mergedSources.Add(source);

                    // log if it came from the command line.
                    if (fromCommandLine)
                    {
                        Log.LogMessageFromResources(MessageImportance.Low, "TrackedVCToolTask.RebuildingSourceCommandLineChanged", source.GetMetadata("FullPath"));
                    }
                }
            }

            return mergedSources.ToArray();
        }

        /// <summary>
        /// Maps the source files to their corresponding command line
        /// </summary>
        protected IDictionary<string, string> MapSourcesToCommandLines()
        {
            IDictionary<string, string> sourcesToCommandLines = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            // This should not ever be reached unless TLogCommandFile is a valid path.
            string tlogPath = TLogCommandFile.GetMetadata("FullPath");

            if (File.Exists(tlogPath))
            {
                using (StreamReader tlog = File.OpenText(tlogPath))
                {
                    bool encounteredInvalidTLogContents = false;
                    string tlogEntry = String.Empty;
                    string tlogLine = tlog.ReadLine();

                    while (tlogLine != null)
                    {
                        if (tlogLine.Length == 0)
                        {
                            encounteredInvalidTLogContents = true;
                            break;
                        }

                        if (tlogLine[0] == '^')
                        {
                            if (tlogLine.Length == 1)
                            {
                                encounteredInvalidTLogContents = true;
                                break;
                            }

                            tlogEntry = tlogLine.Substring(1);
                        }
                        else
                        {
                            string commandLine = null;
                            if (!sourcesToCommandLines.TryGetValue(tlogEntry, out commandLine))
                            {
                                sourcesToCommandLines[tlogEntry] = tlogLine;
                            }
                            else
                            {
                                sourcesToCommandLines[tlogEntry] += "\r\n" + tlogLine;
                            }
                        }

                        tlogLine = tlog.ReadLine();
                    }

                    if (encounteredInvalidTLogContents)
                    {
                        Log.LogWarningWithCodeFromResources("TrackedVCToolTask.RebuildingDueToInvalidTLogContents", tlogPath);
                        sourcesToCommandLines = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                    }
                }
            }
            // If the tlog doesn't exist, that's invalid too, but a message has already been logged, and we
            // wouldn't have even reached this point ...

            return sourcesToCommandLines;
        }

        /// <summary>
        /// Given a table mapping sources to command lines, write that table to disk using
        /// the appropriate tlog name.
        /// </summary>
        /// <param name="sourcesToCommandlines"></param>
        protected void WriteSourcesToCommandLinesTable(IDictionary<string, string> sourcesToCommandLines)
        {
            string tlogFilename = TLogCommandFile.GetMetadata("FullPath");

            // Make sure tlog directory exists
            Directory.CreateDirectory(Path.GetDirectoryName(tlogFilename));

            using (StreamWriter commands = new StreamWriter(tlogFilename, false, System.Text.Encoding.Unicode))
            {
                foreach (KeyValuePair<string, string> sourceAndCommandLinePair in sourcesToCommandLines)
                {
                    commands.WriteLine("^" + sourceAndCommandLinePair.Key);
                    commands.WriteLine(ApplyPrecompareCommandFilter(sourceAndCommandLinePair.Value));
                }
            }
        }

        /// <summary>
        /// Begin the Unicode output before executing the tool if it is needed.
        /// </summary>
        /// <returns></returns>
        public override bool Execute()
        {
            BeginUnicodeOutput();

            bool returnValue = false;

            try
            {
                returnValue = base.Execute();
            }
            finally
            {
                EndUnicodeOutput();
            }

            return returnValue;
        }

        /// <summary>
        /// Executes the tool
        /// </summary>
        /// <param name="pathToTool">The computed path to tool executable on disk</param>
        /// <param name="responseFileCommands">Command line arguments that should go into a temporary response file</param>
        /// <param name="commandLineCommands">Command line arguments that should be passed to the tool executable directly</param>
        /// <returns>exit code from the tool</returns>
        /// <owner>KieranMo</owner>
        protected override int ExecuteTool
            (
            string pathToTool,
            string responseFileCommands,
            string commandLineCommands
            )
        {
            int exitCode = 0;

            if (EnableExecuteTool)
            {
                try
                {
                    // Execute with tracking
                    exitCode = TrackerExecuteTool(pathToTool, responseFileCommands, commandLineCommands);
                }
                finally
                {
                    // Stop using multi-line logging when TrackerExecuteTool is done.
                    // An error needs to be flushed to set log.HasLoggedErrors to true,
                    // to avoid "error code" message from MSBuild.
                    PrintMessage(ParseLine(null), StandardOutputImportanceToUse);

                    if (PostBuildTrackingCleanup)
                        exitCode = PostExecuteTool(exitCode);
                }
            }

            return exitCode;
        }

        /// <summary>
        /// PostExecuteTool will process tlogs to remove duplicates and invalid entires.
        /// </summary>
        /// <param name="exitCode">Exit code from tool execution</param>
        /// <returns>returns exitCode</returns>
        protected virtual int PostExecuteTool(int exitCode)
        {
            if (MinimalRebuildFromTracking || TrackFileAccess)
            {
                // Read input and output tlogs into graphs
                SourceOutputs = new CanonicalTrackedOutputFiles(TLogWriteFiles);
                SourceDependencies = new CanonicalTrackedInputFiles(TLogReadFiles, TrackedInputFiles, ExcludedInputPaths, SourceOutputs, false, MaintainCompositeRootingMarkers);

                // Also generate the source-to-commandline dictionary for use below
                string[] removedMarkersWithSharedOutputs = null;
                IDictionary<string, string> sourcesToCommandLines = MapSourcesToCommandLines();

                if (exitCode != 0)
                {
                    // If the tool errors in some way, we assume that any and all inputs and outputs it wrote during
                    // execution are wrong. So we compact the read and write tlogs to remove the entries for the
                    // set of sources being compiled - the next incremental build will find no entries
                    // and correctly cause the sources to be compiled
                    SourceOutputs.RemoveEntriesForSource(SourcesCompiled);
                    SourceOutputs.SaveTlog();

                    // Compact the read tlog
                    SourceDependencies.RemoveEntriesForSource(SourcesCompiled);
                    SourceDependencies.SaveTlog();

                    if (TrackCommandLines)
                    {
                        // Remove this entry from the command tlog
                        if (MaintainCompositeRootingMarkers)
                        {
                            sourcesToCommandLines.Remove(RootSource);
                        }
                        else
                        {
                            foreach (ITaskItem source in SourcesCompiled)
                            {
                                sourcesToCommandLines.Remove(FileTracker.FormatRootingMarker(source));
                            }
                        }

                        WriteSourcesToCommandLinesTable(sourcesToCommandLines);
                    }
                }
                else
                {
                    // Some tasks have special reasons to add fake outputs
                    // (right now this is only Link -- for the pdb.)
                    AddTaskSpecificOutputs(SourcesCompiled, SourceOutputs);

                    // Some tasks have special reasons to remove specific dependencies
                    // (e.g., MIDL removes dlldata.c, CL removes .i files)
                    RemoveTaskSpecificOutputs(SourceOutputs);

                    // If all went well with the tool execution, then compact the tlogs
                    // to remove any files that are no longer on disk.
                    // This removes any temporary files from the dependency graph
                    // Compact the write tlog
                    SourceOutputs.RemoveDependenciesFromEntryIfMissing(SourcesCompiled);

                    // If any other rooting markers in the dependency graph share our outputs, we want to get rid of them.
                    // This only really makes sense when we're keeping the composite rooting markers around, though.
                    if (MaintainCompositeRootingMarkers)
                    {
                        removedMarkersWithSharedOutputs = SourceOutputs.RemoveRootsWithSharedOutputs(SourcesCompiled);

                        foreach (string rootMarkerToRemove in removedMarkersWithSharedOutputs)
                        {
                            SourceDependencies.RemoveEntryForSourceRoot(rootMarkerToRemove);
                        }
                    }

                    if (TrackedOutputFilesToIgnore != null && TrackedOutputFilesToIgnore.Length > 0)
                    {
                        Dictionary<string, ITaskItem> trackedOutputFilesToRemove = new Dictionary<string, ITaskItem>(StringComparer.OrdinalIgnoreCase);

                        foreach (ITaskItem removeFile in TrackedOutputFilesToIgnore)
                        {
                            // Multiple relative path of could lead to the same full path.  Check for such case.
                            string removeFileFullPath = removeFile.GetMetadata("FullPath").ToUpperInvariant();

                            if (!trackedOutputFilesToRemove.ContainsKey(removeFileFullPath))
                            {
                                trackedOutputFilesToRemove.Add(removeFileFullPath, removeFile);
                            }
                        }

                        // Use an anonymous method to encapsulate the contains check for the tlogs
                        SourceOutputs.SaveTlog(delegate (string fullTrackedPath)
                        {
                            // We need to answer the question "should fullTrackedPath be included in the TLog?"
                            if (trackedOutputFilesToRemove.ContainsKey(fullTrackedPath.ToUpperInvariant()))
                            {
                                return false;
                            }
                            else
                            {
                                return true;
                            }
                        });
                    }
                    else
                    {
                        SourceOutputs.SaveTlog();
                    }

                    DeleteEmptyFile(TLogWriteFiles);

                    RemoveTaskSpecificInputs(SourceDependencies);

                    // Compact the read tlog
                    SourceDependencies.RemoveDependenciesFromEntryIfMissing(SourcesCompiled);
                    if (TrackedInputFilesToIgnore != null && TrackedInputFilesToIgnore.Length > 0)
                    {
                        Dictionary<string, ITaskItem> trackedInputFilesToRemove = new Dictionary<string, ITaskItem>(StringComparer.OrdinalIgnoreCase);

                        foreach (ITaskItem removeFile in TrackedInputFilesToIgnore)
                        {
                            // Multiple relative path of could lead to the same full path.  Check for such case.
                            string removeFileFullPath = removeFile.GetMetadata("FullPath").ToUpperInvariant();

                            if (!trackedInputFilesToRemove.ContainsKey(removeFileFullPath))
                            {
                                trackedInputFilesToRemove.Add(removeFileFullPath, removeFile);
                            }

                        }

                        // Use an anonymous method to encapsulate the contains check for the tlogs
                        SourceDependencies.SaveTlog(delegate (string fullTrackedPath)
                        {
                            // We need to answer the question "should fullTrackedPath be included in the TLog?"
                            if (trackedInputFilesToRemove.ContainsKey(fullTrackedPath))
                            {
                                return false;
                            }
                            else
                            {
                                return true;
                            }
                        });
                    }
                    else
                    {
                        SourceDependencies.SaveTlog();
                    }

                    DeleteEmptyFile(TLogReadFiles);

                    if (TrackCommandLines)
                    {
                        if (MaintainCompositeRootingMarkers)
                        {
                            // If we're maintaining the rooting markers, then we can go ahead and write a single command line --
                            // SourcesCompiled will always contain all sources.
                            string tlogContents = GenerateCommandLine(CommandLineFormat.ForTracking);

                            sourcesToCommandLines[RootSource] = tlogContents;

                            // Make sure that the outputs that we removed from the compact outputs are also removed from the
                            // command line tlog
                            if (removedMarkersWithSharedOutputs != null)
                            {
                                foreach (string rootToRemove in removedMarkersWithSharedOutputs)
                                {
                                    sourcesToCommandLines.Remove(rootToRemove);
                                }
                            }
                        }
                        else
                        {
                            // Otherwise, we want to write a separate command line for each source, so as not to break
                            // minimal rebuild optimization.
                            string sourcesPropertyName = SourcesPropertyName ?? "Sources";
                            string tlogContents = GenerateCommandLineExceptSwitches(new string[] { sourcesPropertyName }, CommandLineFormat.ForTracking);

                            foreach (ITaskItem source in SourcesCompiled)
                            {
                                sourcesToCommandLines[FileTracker.FormatRootingMarker(source)] = tlogContents + " " + source.GetMetadata("FullPath").ToUpperInvariant();
                            }
                        }

                        WriteSourcesToCommandLinesTable(sourcesToCommandLines);
                    }
                }
            }

            return exitCode;
        }

        /// <summary>
        /// Can be overridden if a task wants to remove specific outputs from the output graph
        /// after the tool has executed. CL uses this for preprocessor outputs.
        /// </summary>
#if WHIDBEY_VISIBILITY
        internal
#else
        protected
#endif
        virtual void RemoveTaskSpecificOutputs(CanonicalTrackedOutputFiles compactOutputs)
        {
            // Do nothing, normally
        }

        /// <summary>
        /// Can be overridden if a task wants to remove specific outputs from the output graph
        /// after the tool has executed. CL uses this for preprocessor outputs.
        /// </summary>
#if WHIDBEY_VISIBILITY
        internal
#else
        protected
#endif
        virtual void RemoveTaskSpecificInputs(CanonicalTrackedInputFiles compactInputs)
        {
            // Do nothing, normally
        }

        /// <summary>
        /// Can be overridden if a task has outputs they want to add to the dependency
        /// graph.  Link uses this for the PDB.
        /// </summary>
#if WHIDBEY_VISIBILITY
        internal
#else
        protected
#endif
        virtual void AddTaskSpecificOutputs(ITaskItem[] sources, CanonicalTrackedOutputFiles compactOutputs)
        {
            // Do nothing, normally
        }

        /// <summary>
        /// Logs the tool name and the path from where it is being run.
        /// </summary>
        /// <remarks>
        /// Overriding to log the path to the tool, not to the tracker
        /// (if the tracker is being used).
        /// </remarks>
        /// <owner>danmose</owner>
        override protected void LogPathToTool
        (
            string toolName,
            string pathToTool
        )
        {
            // Note that we're passing ResolvedPathToTool here, instead of pathToTool, to log
            // the actual tool's path instead of the tracker's path
            base.LogPathToTool(toolName, ResolvedPathToTool);
        }

        /// <summary>
        /// Executes the tool possibly with file tracking.
        /// </summary>
        /// <param name="pathToTool">The computed path to tool executable on disk</param>
        /// <param name="responseFileCommands">Command line arguments that should go into a temporary response file</param>
        /// <param name="commandLineCommands">Command line arguments that should be passed to the tool executable directly</param>
        /// <returns>exit code from the tool</returns>
        /// <owner>KieranMo</owner>
        protected int TrackerExecuteTool
            (
            string pathToTool,
            string responseFileCommands,
            string commandLineCommands
            )
        {
            string commandLineArgs;
            string commandLineTool;
            string fileTrackerDllPath = null;
            string trackerResponseFile = null;
            bool trackFileAccess = TrackFileAccess;

            //Expand the envVariable for the input values
            string pathToToolExpanded = System.Environment.ExpandEnvironmentVariables(pathToTool);
            string responseFileCommandsExpanded = responseFileCommands;
            string commandLineCommandsExpanded = System.Environment.ExpandEnvironmentVariables(commandLineCommands);

            try
            {
                if (trackFileAccess)
                {
                    ExecutableType toolArchitecture = ExecutableType.SameAsCurrentProcess;

                    // ToolArchitecture passed in from the targets wins, next is ToolType defined by the task owners,
                    // and if neither is specified, we default to ExecutableType.SameAsCurrentProcess.
                    if (!String.IsNullOrEmpty(ToolArchitecture))
                    {
                        if (!Enum.TryParse<ExecutableType>(ToolArchitecture, out toolArchitecture))
                        {
                            Log.LogErrorWithCodeFromResources("General.InvalidValue", "ToolArchitecture", this.GetType().Name);
                            return -1;
                        }
                    }
                    else if (ToolType.HasValue)
                    {
                        toolArchitecture = ToolType.Value;
                    }

                    if (toolArchitecture == ExecutableType.Native32Bit || toolArchitecture == ExecutableType.Native64Bit)
                    {
                        // Tool is native, get the data for the actual tool exe without guessing.
                        // Look up can sometimes fails, use the suggested ToolArchitecture.
                        bool is64bit;
                        if (NativeMethodsShared.Is64bitApplication(pathToToolExpanded, out is64bit))
                        {
                            toolArchitecture = is64bit ? ExecutableType.Native64Bit : ExecutableType.Native32Bit;
                        }
                    }

                    try
                    {
                        commandLineTool = FileTracker.GetTrackerPath(toolArchitecture, TrackerSdkPath);

                        if (commandLineTool == null)
                        {
                            Log.LogErrorFromResources("Error.MissingFile", "tracker.exe");
                        }
                    }
                    catch (Exception e)
                    {
                        if (ExceptionHandling.NotExpectedException(e))
                        {
                            throw;
                        }

                        Log.LogErrorWithCodeFromResources("General.InvalidValue", "TrackerSdkPath", this.GetType().Name);
                        return -1;
                    }

                    try
                    {
                        fileTrackerDllPath = FileTracker.GetFileTrackerPath(toolArchitecture, TrackerFrameworkPath);

                        // if not found, tracker.exe will look for it by itself
                    }
                    catch (Exception e)
                    {
                        if (ExceptionHandling.NotExpectedException(e))
                        {
                            throw;
                        }

                        Log.LogErrorWithCodeFromResources("General.InvalidValue", "TrackerFrameworkPath", this.GetType().Name);
                        return -1;
                    }
                }
                else
                {
                    commandLineTool = pathToToolExpanded;
                }

                if (!String.IsNullOrEmpty(commandLineTool))
                {
                    // pathToToolExpanded was already made into a full path by ComputePathToTool(), and the
                    // two FileTracker functions also return full paths; if we make it to this point with an
                    // unrooted path, there's something strange going on.
                    ErrorUtilities.VerifyThrowInternalRooted(commandLineTool);

                    if (trackFileAccess)
                    {
                        // Log the tracking command (the command line beginning with 'tracker.exe') at low priority, because it's an
                        // implementation detail; we log the command the user expects (omitting tracker) in <see cref="LogPathToTool"/>.
                        string logCommandLineArgs = FileTracker.TrackerArguments(pathToToolExpanded, commandLineCommandsExpanded, fileTrackerDllPath, TrackerIntermediateDirectory, RootSource, CancelEventName);

                        Log.LogMessageFromResources(MessageImportance.Low, "Native_TrackingCommandMessage");

                        string message = commandLineTool + (AttributeFileTracking ? " /a " : " ")
                                                            + (TrackReplaceFile ? "/f " : "")
                                                            + logCommandLineArgs + " " + responseFileCommandsExpanded;

                        Log.LogMessage(MessageImportance.Low, message);

                        // Put all the parameters into a temporary response file so we don't have to worry
                        // about how long the command-line is going to be

                        // May throw IO-related exceptions
                        trackerResponseFile = FileUtilities.GetTemporaryFile();

                        using (StreamWriter responseFileStream = new StreamWriter(trackerResponseFile, false, Encoding.Unicode))
                        {
                            responseFileStream.Write(FileTracker.TrackerResponseFileArguments(fileTrackerDllPath, TrackerIntermediateDirectory, RootSource, CancelEventName));
                        }

                        commandLineArgs = (AttributeFileTracking ? "/a @\"" : "@\"")
                                            + trackerResponseFile + "\""
                                            + (TrackReplaceFile ? " /f " : "")
                                            + FileTracker.TrackerCommandArguments(pathToToolExpanded, commandLineCommandsExpanded);
                    }
                    else
                    {
                        commandLineArgs = commandLineCommandsExpanded;
                    }

                    return base.ExecuteTool(commandLineTool, responseFileCommandsExpanded, commandLineArgs);
                }
                else
                {
                    // Did not find the tool to run: whatever searched for it should have already logged an error
                    return -1;
                }
            }
            finally
            {
                if (trackerResponseFile != null)
                {
                    DeleteTempFile(trackerResponseFile);
                }
            }
        }

        /// <summary>
        /// If set, replaces the value of the PATH environment variable with this value,
        /// just for the process of the tool that will be launched
        /// </summary>
        public string PathOverride
        {
            set
            {
                pathOverride = value;
            }
            get
            {
                return pathOverride;
            }
        }

        private string pathOverride;

        /// <summary>
        /// Gets whether Unicode output will be used when tool executes.
        /// By default, Unicode output won't be used.
        /// </summary>
        protected virtual bool UseUnicodeOutput
        {
            get { return false; }
        }

        #region Logging and Piping

        /// <summary>
        /// Used for reading Unicode pipe.
        /// </summary>
        private SafeFileHandle unicodePipeReadHandle;

        /// <summary>
        /// Used to writing Unicode pipe.
        /// </summary>
        private SafeFileHandle unicodePipeWriteHandle;

        /// <summary>
        /// Used for signalling when the Unicode output ends.
        /// </summary>
        private AutoResetEvent unicodeOutputEnded;

        /// <summary>
        /// Begin the Unicode output if the tool specifies UseUnicodeOutput to true.
        /// When Unicode output starts, an additional pipe will be created to facilitate the output,
        /// and then an environment variable in the form of "VS_UNICODE_OUTPUT=xxxx" will be passed
        /// into the newly created child process (xxxx is the int value of the pipe write handle).
        /// </summary>
        private void BeginUnicodeOutput()
        {
            unicodePipeReadHandle = null;
            unicodePipeWriteHandle = null;
            unicodeOutputEnded = null;

            if (UseUnicodeOutput)
            {
                NativeMethodsShared.SecurityAttributes securityAttributes = new NativeMethodsShared.SecurityAttributes();
                securityAttributes.lpSecurityDescriptor = NativeMethodsShared.NullIntPtr;
                securityAttributes.bInheritHandle = true;

                if (NativeMethodsShared.CreatePipe(out unicodePipeReadHandle, out unicodePipeWriteHandle, securityAttributes, 0))
                {
                    List<string> environmentVariables = new List<string>();
                    if (EnvironmentVariables != null)
                    {
                        environmentVariables.AddRange(EnvironmentVariables);
                    }
                    environmentVariables.Add("VS_UNICODE_OUTPUT=" + unicodePipeWriteHandle.DangerousGetHandle());

                    if (StandardOutputEncoding == Encoding.UTF8)
                    {
                        environmentVariables.Add("TRACKER_CONSOLEPAGE=65001");
                    }

                    EnvironmentVariables = environmentVariables.ToArray();

                    unicodeOutputEnded = new AutoResetEvent(false);
                    ThreadPool.QueueUserWorkItem(new WaitCallback(ReadUnicodeOutput));
                }
                else
                {
                    Log.LogWarningWithCodeFromResources("TrackedVCToolTask.CreateUnicodeOutputPipeFailed", ToolName);
                }
            }
            else
            {
                List<string> environmentVariables = new List<string>();
                if (EnvironmentVariables != null)
                {
                    environmentVariables.AddRange(EnvironmentVariables);
                }
                if (StandardOutputEncoding == Encoding.UTF8)
                {
                    environmentVariables.Add("TRACKER_CONSOLEPAGE=65001");
                }
                EnvironmentVariables = environmentVariables.ToArray();
            }
        }

        /// <summary>
        /// End the Unicode output and release all the resources.
        /// </summary>
        private void EndUnicodeOutput()
        {
            if (UseUnicodeOutput)
            {
                if (unicodePipeWriteHandle != null)
                {
                    unicodePipeWriteHandle.Close();
                }

                if (unicodeOutputEnded != null)
                {
                    unicodeOutputEnded.WaitOne();
                    unicodeOutputEnded.Close();
                }

                if (unicodePipeReadHandle != null)
                {
                    unicodePipeReadHandle.Close();
                }
            }
        }

        /// <summary>
        /// ToolTask class calls ProcessStarted right after it starts the tool process.
        /// </summary>
        protected override void ProcessStarted()
        {
            if (UseUnicodeOutput)
            {
                // We must close write handle of the pipe we created right after the tool process is started to avoid "build cancelled but cannot finish" problem.
                if (unicodePipeWriteHandle != null)
                {
                    unicodePipeWriteHandle.Close();
                    unicodePipeWriteHandle = null;
                }
            }
        }

        private static readonly char[] NewlineArray = Environment.NewLine.ToCharArray();

        /// <summary>
        /// Read out and display the Unicode pipe contents which are written by the tool process.
        /// </summary>
        /// <param name="stateInfo"></param>
        private void ReadUnicodeOutput(object stateInfo)
        {
            const int bufferSize = 1024;
            byte[] buffer = new byte[bufferSize];
            uint bytesRead;
            StringBuilder sbRemainder = new StringBuilder(bufferSize);

            while (true)
            {
                if (!NativeMethodsShared.ReadFile(unicodePipeReadHandle, buffer, bufferSize, out bytesRead, NativeMethodsShared.NullIntPtr) || bytesRead == 0)
                {
                    if (sbRemainder.Length > 0)
                    {
                        LogEventsFromTextOutput(sbRemainder.ToString(), StandardOutputImportanceToUse);
                    }
                    break;
                }

                sbRemainder.Append(Encoding.Unicode.GetString(buffer, 0, (int)bytesRead));
                
                int index = 0;
                while (index < sbRemainder.Length)
                {
                    // search for \r\n
                    if (sbRemainder[index] == '\n' || sbRemainder[index] == '\r')
                    {
                        string line = sbRemainder.ToString(0, index+1).Trim(NewlineArray);
                        sbRemainder.Remove(0, index+1);
                        index = 0;
                        if (line.Length > 0)
                            LogEventsFromTextOutput(line, StandardOutputImportanceToUse);
                    }

                    index++;
                }
            }

            unicodeOutputEnded.Set();
        }

        #endregion

        private static readonly Regex extraNewlineRegex = new Regex(@"(\r?\n)?(\r?\n)+");

        /// <summary>
        /// Overwrite this function to apply modification to the string before it compares
        /// </summary>
        /// <param name="value">the string to be modified</param>
        /// <return>returns the unmodified string</return>
        public virtual string ApplyPrecompareCommandFilter(string value)
        {
            // Reduce multiple newlines to a single newline
            return extraNewlineRegex.Replace(value, "$2");
        }

        /// <summary>
        /// Helper function to remove a switch from an command line
        /// </summary>
        /// <param name="removalWord">the string to remove</param>
        /// <param name="cmdString">the command line</param>
        /// <param name="removeMultiple">Remove the first one found or remove multiple. Default remove only first one found.</param>
        /// <return>returns the new command line</return>
        public static string RemoveSwitchFromCommandLine(string removalWord, string cmdString, bool removeMultiple = false)
        {
            int index = 0;
            while ((index = cmdString.IndexOf(removalWord, index, StringComparison.Ordinal)) >= 0)
            {
                int indexEnd;

                // if start of the string or a space exist before it.  Ensure it is standalone switch
                if (index == 0 || cmdString[index - 1] == ' ')
                {
                    indexEnd = cmdString.IndexOf(' ', index);
                    if (indexEnd >= 0) //include the space if more switches are to follow
                    {
                        indexEnd++;
                    }
                    else  //didn't find the end value, set to end position and trim to previous space
                    {
                        indexEnd = cmdString.Length;
                        index--;
                    }
                    cmdString = cmdString.Remove(index, indexEnd - index);

                    if (!removeMultiple)
                        break;
                }

                // increment next index.  If the is pass the length of the string, then break the loop.
                index++;

                if (index >= cmdString.Length)
                {
                    break;
                }
            }

            return cmdString;
        }

        /// <summary>
        /// Helper function to delete files
        /// </summary>
        /// <param name="filesToDelete">List of files</param>
        /// <return>returns number of files deleted</return>
        protected static int DeleteFiles(ITaskItem[] filesToDelete)
        {
            if (filesToDelete == null)
            {
                return 0;
            }

            ITaskItem[] tlogFiles = TrackedDependencies.ExpandWildcards(filesToDelete);

            if (tlogFiles.Length == 0)
            {
                return 0;
            }

            int count = 0;

            foreach (ITaskItem item in tlogFiles)
            {
                try
                {
                    FileInfo info = new FileInfo(item.ItemSpec);

                    if (!info.Exists)  // if the file doesn't exist, then move on
                    {
                        continue;
                    }

                    info.Delete();
                    ++count;
                }
                catch (Exception ex)
                {
                    if ((ex is System.Security.SecurityException) ||
                    (ex is ArgumentException) ||
                    (ex is UnauthorizedAccessException) ||
                    (ex is PathTooLongException) ||
                    (ex is NotSupportedException))
                    {
                        continue;
                    }
                    throw;
                }
            }

            return count;
        }

        /// <summary>
        /// Helper function to delete empty files (sometimes files contain type information which makes them non-zero size.)
        /// </summary>
        /// <param name="filesToDelete">List of files</param>
        /// <return>returns number of files deleted</return>
        protected static int DeleteEmptyFile(ITaskItem[] filesToDelete)
        {
            if (filesToDelete == null)
            {
                return 0;
            }

            ITaskItem[] tlogFiles = TrackedDependencies.ExpandWildcards(filesToDelete);

            if (tlogFiles.Length == 0)
            {
                return 0;
            }

            int count = 0;

            foreach (ITaskItem item in tlogFiles)
            {
                bool delete = false;

                try
                {
                    FileInfo info = new FileInfo(item.ItemSpec);

                    if (!info.Exists)  // if the file doesn't exist, then move on
                    {
                        continue;
                    }

                    if (info.Length <= 4 /* 32bit */)
                    {
                        delete = true;
                    }

                    if (delete)
                    {
                        info.Delete();
                        ++count;
                    }
                }
                catch (Exception ex)
                {
                    if ((ex is System.Security.SecurityException) ||
                    (ex is ArgumentException) ||
                    (ex is UnauthorizedAccessException) ||
                    (ex is PathTooLongException) ||
                    (ex is NotSupportedException))
                    {
                        continue;
                    }
                    throw;
                }
            }

            return count;
        }

    }
}
