namespace Microsoft.Build.CPPTasks
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Text;
    using System.Text.RegularExpressions;
    using Microsoft.Build.Utilities;
    using Microsoft.Build.Framework;
    using Microsoft.Build.Shared;

    /// <summary>
    /// The Clang task to wrap the Clang.exe tool
    /// </summary>
    public partial class ClangCompile : TrackedVCToolTask
    {
        private ITaskItem[] preprocessOutput = new ITaskItem[0];

        string firstReadTlog = typeof(ClangCompile).FullName + ".read.1.tlog";

        string firstWriteTlog = typeof(ClangCompile).FullName + ".write.1.tlog";

        // gcc format:
        // <filename>:<line number>:<column number> : error: message
        static Regex gccMessageRegex = new Regex(@"^\s*(?<FILENAME>[^:]*):(?<LINE>\d*):(?<COLUMN>\d*)\s*:\s*(?<CATEGORY>fatal error|error|warning|note):(?<TEXT>.*)",
                                         RegexOptions.Compiled | RegexOptions.IgnoreCase);

        public bool GNUMode { get; set; }
        public string ClangVersion { get; set; }

        /// <summary>
        /// ClangComile can only handle one compile at a time, so it is safe
        /// to leave CompositeRootingMarkers to the same value (true) as clanglink,
        /// because the Canonical Tracking caches MaintainCompositeRootingMarkers in to its dependency table.
        /// </summary>
        protected override bool MaintainCompositeRootingMarkers
        {
            get { return false; }
        }

        /// <summary>
        /// Names of the read tlogs for this tool
        /// </summary>
        protected override string[] ReadTLogNames
        {
            get
            {
                string toolName = Path.GetFileNameWithoutExtension(ToolExe);
                return new string[] { firstReadTlog, toolName + ".read.*.tlog", toolName + ".*.read.*.tlog", toolName + "-*.read.*.tlog",
                    toolName + ".delete.*.tlog", toolName + ".*.delete.*.tlog", toolName + "-*.delete.*.tlog"};
            }
        }

        /// <summary>
        /// Names of the write tlogs for this tool
        /// </summary>
        protected override string[] WriteTLogNames
        {
            get
            {
                string toolName = Path.GetFileNameWithoutExtension(ToolExe);
                return new string[] { firstWriteTlog, toolName + ".write.*.tlog", toolName + ".*.write.*.tlog", toolName + "-*.write.*.tlog" };
            }
        }

        /// <summary>
        /// Name of the commandline tlog for this tool
        /// </summary>
        protected override string CommandTLogName
        {
            get
            {
                string toolName = Path.GetFileNameWithoutExtension(ToolExe);
                return toolName + ".command.1.tlog";
            }
        }

        /// <summary>
        /// Accessor for the abstract base class. This can't be concrete in the base class
        /// because TrackerLogDirectory cannot be moved into the base class because
        /// it needs to be specified in the XML task definition (corresponding to the other half of
        /// this partial class) because some other XML-specified properties use it as a fallback value.
        /// The XML task definition won't spit the "override" keyword, so we can't put an abstract
        /// TrackerLogDirectory in the base class either.
        /// <returns>TrackerLogDirectory</returns>
        protected override string TrackerIntermediateDirectory
        {
            get
            {
                if (TrackerLogDirectory != null)
                {
                    return TrackerLogDirectory;
                }
                else
                {
                    return String.Empty;
                }
            }
        }

        /// <summary>
        /// Accessor for the abstract base class, as Sources cannot be moved into the base
        /// class as it needs to be specified in the XML task definition. More importantly, it is task
        /// specific in that it can either be Source or Sources depending on the tool.
        /// </summary>
        /// <returns></returns>
        protected override ITaskItem[] TrackedInputFiles
        {
            get
            {
                return Sources;
            }
        }

        /// <summary>
        /// Turn on tracking for ReplaceFile API.
        /// </summary>
        protected override bool TrackReplaceFile
        {
            get
            {
                // For LLVM/clang versions 5 and earlier, it uses ReplaceFile when writing to an output file.
                // In clang 6+ it uses CreateFile, however it's safe to still turn on tracking for ReplaceFile
                // as it will not be called, unless clang or the kernal library changes in the future.
                // See bug# 649451 for details.
                return true;
            }
        }

        protected override void RemoveTaskSpecificInputs(CanonicalTrackedInputFiles compactInputs)
        {
            // Remove PCH file from dependancy of the current input CanonicalTrackedInput
            // Only remove during create mode
            if (IsPropertySet("PrecompiledHeader") && PrecompiledHeader != "Create")
            {
                return;
            }

            string pchFileName;

            if (IsPropertySet("ObjectFileName"))
            {
                pchFileName = ObjectFileName;
            }
            else
            {
                return;  //hopeless
            }

            TaskItem taskItemPch = new TaskItem(pchFileName);

            compactInputs.RemoveDependencyFromEntry(Sources, taskItemPch);
        }

        /// <summary>
        /// Override ExecuteTool to print out the compiling files
        /// </summary>
        protected override int ExecuteTool(string pathToTool, string responseFileCommands, string commandLineCommands)
        {
            foreach (var sourcesCompiled in SourcesCompiled)
            {
                Log.LogMessage(MessageImportance.High, Path.GetFileName(sourcesCompiled.ItemSpec));
            }

            // ClangCompile tlogs conflicts with ClangLink, as a workaround, hardcode a tlog that is different
            // and is the first in ReadTLogNames and WriteTLogNames
            int counter = 0;

            do
            {
              counter++;

              if (!File.Exists(Path.Combine(TrackerIntermediateDirectory, firstReadTlog)))
              {
                  try
                  {
                      using (File.Create(Path.Combine(TrackerIntermediateDirectory, firstReadTlog))) { }
                  }
                  catch (IOException)
                  {
                      System.Threading.Thread.Sleep(50);
                      continue;
                  }
              }

              if (!File.Exists(Path.Combine(TrackerIntermediateDirectory, firstWriteTlog)))
              {
                  try
                  {
                      using (File.Create(Path.Combine(TrackerIntermediateDirectory, firstWriteTlog))) { }
                  }
                  catch (IOException)
                  {
                      System.Threading.Thread.Sleep(50);
                      continue;
                  }
              }

              break;
            } while (counter < 30);

            if (this.GNUMode)
            {
                errorListRegexList.Add(gccMessageRegex);
            }

            return base.ExecuteTool(pathToTool, responseFileCommands, commandLineCommands);
        }

        /// <summary>
        /// Override encoding for GNU
        /// </summary>
        /// <remarks>GNU only supports UTF8 response file without BOM.</remarks>
        protected override Encoding ResponseFileEncoding
        {
            get
            {
                return GNUMode ? new UTF8Encoding(false) : base.ResponseFileEncoding;
            }
        }

        /// <summary>
        /// Also escape Slash Space
        /// </summary>
        /// <returns>Response file commands</returns>
        protected override string GenerateResponseFileCommandsExceptSwitches(string[] switchesToRemove, CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.EscapeTrailingSlash)
        {
            string commandString = base.GenerateResponseFileCommandsExceptSwitches(switchesToRemove, format, EscapeFormat.EscapeTrailingSlash);

            if (format == CommandLineFormat.ForBuildLog)
            {
                commandString = commandString.Replace(@"\", @"\\");

                // '\\ ' => '\\\\ ' slash space becomes slash slash space, but that is too much, revert this case
                commandString = commandString.Replace(@"\\\\ ", @"\\ ");
            }

            return commandString;
        }

        protected override Encoding StandardOutputEncoding
        {
            get
            {
                return Encoding.UTF8;
            }
        }

        protected override Encoding StandardErrorEncoding
        {
            get
            {
                return Encoding.UTF8;
            }
        }
    }
}
