using System;
using System.Globalization;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Runtime.InteropServices;
using Microsoft.Build.Tasks;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;
using System.Collections.Specialized;
using System.Resources;
using Microsoft.Build.Shared;

namespace Microsoft.Build.CPPTasks
{
    /// <summary>
    /// The base class of all VC tool tasks that have been generated from XML.
    /// </summary>
    public abstract class VCToolTask : ToolTask
    {
        #region Enums
        /// <summary>
        /// Determines whether or not a command line should be used for the build itself or for a response file.
        /// By making this distinction we can convert file and directory names to upper case for response files.
        /// This will prevent case sensitivity issues when comparing the current command line to the one in the response
        /// file.
        /// </summary>
        /// <owner>JamesJ</owner>
        public enum CommandLineFormat
        {
            /// <summary>
            /// Use this command line for the build log. This is the command line that is pass to the specified tool
            /// </summary>
            ForBuildLog,

            /// <summary>
            /// Use this command line for writing to the response file for tracking to determine out-of-date commands
            /// lines for the next build
            /// </summary>
            ForTracking
        }

        /// <summary>
        /// Additional Escape Options
        /// </summary>
        [Flags]
        public enum EscapeFormat
        {
            /// <summary>
            /// Default behavior
            /// </summary>
            Default = 0,

            /// <summary>
            /// Escape the trailing slash
            /// </summary>
            EscapeTrailingSlash = 1
        }
        #endregion

        #region Private Members

        private Dictionary<string, ToolSwitch> activeToolSwitchesValues = new Dictionary<string, ToolSwitch>();
        private IntPtr cancelEvent;
        private string cancelEventName;
        private bool fCancelled;

        /// <summary>
        /// The dictionary that holds all set switches
        /// The string is the name of the property, and the ToolSwitch holds all of the relevant information
        /// i.e., switch, boolean value, type, etc.
        /// </summary>
        private Dictionary<string, ToolSwitch> activeToolSwitches = new Dictionary<string, ToolSwitch>(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// The dictionary holds all of the legal values that are associated with a certain switch.
        /// For example, the key Optimization would hold another dictionary as the value, that had the string pairs
        /// "Disabled", "/Od"; "MaxSpeed", "/O1"; "MinSpace", "/O2"; "Full", "/Ox" in it.
        /// </summary>
        private Dictionary<string, Dictionary<string, string>> values = new Dictionary<string, Dictionary<string, string>>(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Any additional options (as a literal string) that may have been specified in the project file
        /// We eventually want to get rid of this
        /// </summary>
        private string additionalOptions = String.Empty;

        /// <summary>
        /// The prefix to append before all switches
        /// </summary>
        private char prefix = '/';

        /// <summary>
        /// The private log
        /// </summary>
        private TaskLoggingHelper logPrivate;

        #endregion // Private Members

        #region Constructors

        protected VCToolTask(ResourceManager taskResources)
            : base(taskResources)
        {
            cancelEventName = "MSBuildConsole_CancelEvent" + Guid.NewGuid().ToString("N");
            cancelEvent = VCTaskNativeMethods.CreateEventW(IntPtr.Zero, false /* resets automatically after one waiting thread has been released */, false /* initial state is not signaled */, cancelEventName);
            fCancelled = false;

            logPrivate = new TaskLoggingHelper(this);
            logPrivate.TaskResources = AssemblyResources.PrimaryResources;
            logPrivate.HelpKeywordPrefix = "MSBuild.";

            IgnoreUnknownSwitchValues = false;
        }

        #endregion // Constructors

        #region Properties

        /// <summary>
        /// The list of all the switches that have been set
        /// </summary>
        protected Dictionary<string, ToolSwitch> ActiveToolSwitches
        {
            get
            {
                return activeToolSwitches;
            }
        }

        /// <summary>
        /// The additional options that have been set. These are raw switches that
        /// go last on the command line.
        /// </summary>
        public string AdditionalOptions
        {
            get
            {
                return additionalOptions;
            }
            set
            {
                additionalOptions = TranslateAdditionalOptions(value);
            }
        }

        /// <summary>
        /// Tasks can implement this method if they need to modify additional options somehow.
        /// </summary>
        protected virtual string TranslateAdditionalOptions(string options)
        {
            return options;
        }

        /// <summary>
        /// Overridden to use UTF16, which works better than UTF8 for older versions of CL, LIB, etc.
        /// </summary>
        /// <comments>
        /// See VSWhidbey #368228.
        /// </comments>
        protected override Encoding ResponseFileEncoding
        {
            get
            {
                return Encoding.Unicode;
            }
        }

        /// <summary>
        /// Ordered list of switches
        /// </summary>
        /// <returns>ArrayList of switches in declaration order</returns>
        protected virtual ArrayList SwitchOrderList
        {
            get
            {
                return null;
            }
        }

        ///<summary>
        /// Returns the name of the cancel event that was created by this task.
        /// Protected so that it can be accessed by tracked VC tasks that derive from
        /// this task.
        ///</summary>
        protected string CancelEventName
        {
            get { return cancelEventName; }
        }

        ///<summary>
        /// Returns the private log
        ///</summary>
        protected TaskLoggingHelper LogPrivate
        {
            get { return logPrivate; }
        }

        /// <summary>
        /// Importance with which to log text from in the standard out stream.
        /// We make this high imporance because we want, e.g. the messages that CL outputs
        /// when it finishes compiling a file to be visible even on minimal verbosity.
        /// </summary>
        protected override MessageImportance StandardOutputLoggingImportance
        {
            get { return MessageImportance.High; }
        }

        /// <summary>
        /// Importance with which to log text from the standard error stream.
        /// </summary>
        /// <remarks>
        /// Made "High" because stdout is also high importance.
        /// </remarks>
        protected override MessageImportance StandardErrorLoggingImportance
        {
            get { return MessageImportance.High; }
        }

        /// <summary>
        /// The string that is always appended on the command line. Overridden by deriving classes.
        /// </summary>
        protected virtual string AlwaysAppend
        {
            get
            {
                return String.Empty;
            }
            set
            {
                // do nothing
            }
        }

        /// <summary>
        /// Set this string as Regex property
        /// </summary>
        public ITaskItem[] ErrorListRegex
        {
            set
            {
                foreach (ITaskItem regx in value)
                {
                    errorListRegexList.Add(new Regex(regx.ItemSpec, RegexOptions.Compiled | RegexOptions.IgnoreCase, TimeSpan.FromMilliseconds(100)));
                }
            }
        }

        /// <summary>
        /// Set this string as Regex property
        /// </summary>
        public ITaskItem[] ErrorListListExclusion
        {
            set
            {
                foreach (ITaskItem regx in value)
                {
                    errorListRegexListExclusion.Add(new Regex(regx.ItemSpec, RegexOptions.Compiled | RegexOptions.IgnoreCase, TimeSpan.FromMilliseconds(100)));
                }
            }
        }

        /// <summary>
        /// Enable or disable Multiline Errorlist
        /// </summary>
        public bool EnableErrorListRegex { get; set; } = true;

        /// <summary>
        /// Force singleline within Multiline Errorlist
        /// </summary>
        /// <remarks>Use the Multiline Errorlist system but force each line as its own entry.</remarks>
        public bool ForceSinglelineErrorListRegex { get; set; } = false;

        /// <summary>
        /// The non-zero exit codes, if any, that it is acceptable for the wrapped tool to return.
        /// </summary>
        public virtual string[] AcceptableNonzeroExitCodes { get; set; }

        public Dictionary<string, ToolSwitch> ActiveToolSwitchesValues
        {
            get
            {
                return activeToolSwitchesValues;
            }

            set
            {
                activeToolSwitchesValues = value;
            }
        }

        #endregion // Properties

        #region ToolTask Members

        /// <summary>
        /// Task yielding can change working directory, set this to allow the tool to run from a specific directory
        /// </summary>
        public string EffectiveWorkingDirectory { get; set; }

        /// <summary>
        /// Returns the resolved path to the tool
        /// </summary>
        [Output]
        public string ResolvedPathToTool { get; protected set; }

        /// <summary>
        /// Called from base ToolTask to get the Workding Directory.
        /// </summary>
        /// <returns>Path to desire working directory</returns>
        /// <remarks>Safe to return return null.</remarks>
        protected override string GetWorkingDirectory()
        {
            return EffectiveWorkingDirectory;
        }

        /// <summary>
        /// This method is called to find the tool if ToolPath wasn't specified.
        /// We just return the name of the tool so it can be found on the path.
        /// Deriving classes can choose to do something else.
        /// </summary>
        protected override string GenerateFullPathToTool()
        {
            return ToolName;
        }

        /// <summary>
        /// Validates all of the set properties that have either a string type or an integer type
        /// </summary>
        /// <returns></returns>
        override protected bool ValidateParameters()
        {
            return !logPrivate.HasLoggedErrors && !Log.HasLoggedErrors;
        }

        /// <summary>
        /// Generates the command line for this task.
        /// </summary>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        /// <returns>Command line</returns>
        public string GenerateCommandLine(CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.Default)
        {
            string commandLineCommands = GenerateCommandLineCommands(format, escapeFormat);
            string responseFileCommands = GenerateResponseFileCommands(format, escapeFormat);

            return String.IsNullOrEmpty(commandLineCommands) ? responseFileCommands : commandLineCommands + " " + responseFileCommands;
        }

        /// <summary>
        /// Generates the command line for this task, but removes the indicated
        /// switches from the command line if they would otherwise have been added
        /// to it.
        /// IMPORTANT NOTE:  There is no support for removing switches that are
        /// part of the command-line only commands; only for switches that may go
        /// in the response file.
        /// </summary>
        /// <param name="switchesToRemove">Array of the names of switches that should
        /// not show up in the generated command line.</param>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        /// <returns>Command line without switches that have been specified as switches to remove</returns>
        public string GenerateCommandLineExceptSwitches(string[] switchesToRemove, CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.Default)
        {
            string commandLineCommands = GenerateCommandLineCommandsExceptSwitches(switchesToRemove, format, escapeFormat);
            string responseFileCommands = GenerateResponseFileCommandsExceptSwitches(switchesToRemove, format, escapeFormat);

            return String.IsNullOrEmpty(commandLineCommands) ? responseFileCommands : commandLineCommands + " " + responseFileCommands;
        }

        virtual protected string GenerateCommandLineCommandsExceptSwitches(string[] switchesToRemove, CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.Default)
        {
            return string.Empty;
        }

        /// <summary>
        /// Creates the command line and returns it as a string by:
        /// 1. Adding all switches with the default set to the active switch list
        /// 2. Customizing the active switch list (overridden in derived classes)
        /// 3. Iterating through the list and appending switches
        /// </summary>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        /// <returns>Command line</returns>
        protected override string GenerateResponseFileCommands()
        {
            return GenerateResponseFileCommands(CommandLineFormat.ForBuildLog, EscapeFormat.Default);
        }

        /// <summary>
        /// Creates the command line and returns it as a string by:
        /// 1. Adding all switches with the default set to the active switch list
        /// 2. Customizing the active switch list (overridden in derived classes)
        /// 3. Iterating through the list and appending switches
        /// </summary>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        /// <returns>Command line</returns>
        virtual protected string GenerateResponseFileCommands(CommandLineFormat format, EscapeFormat escapeFormat)
        {
            return GenerateResponseFileCommandsExceptSwitches(new string[] { }, format, escapeFormat);
        }

        /// <summary>
        /// Creates the command line and returns it as a string.
        /// </summary>
        /// <returns>Command line</returns>
        protected override string GenerateCommandLineCommands()
        {
            return GenerateCommandLineCommands(CommandLineFormat.ForBuildLog, EscapeFormat.Default);
        }

        /// <summary>
        /// Creates the command line and returns it as a string.
        /// The default behavior of this method is to return an empty string and needs to be overridden in derived classes
        /// in order to create command line command output.
        /// </summary>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        /// <returns>Resulting command line</returns>
        virtual protected string GenerateCommandLineCommands(CommandLineFormat format, EscapeFormat escapeFormat)
        {
            return GenerateCommandLineCommandsExceptSwitches(new string[] { }, format, escapeFormat);
        }

        /// <summary>
        /// Generates the portion of the command line for this task that can go into
        /// the response file, but removes the indicated switches from the command line
        /// if they would otherwise have been added to it.
        /// </summary>
        /// <param name="switchesToRemove">Array of the names of switches that should
        /// not show up in the generated command line.</param>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        virtual protected string GenerateResponseFileCommandsExceptSwitches(string[] switchesToRemove, CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.Default)
        {
            bool hasSetAdditionalOptions = false;

            // the next three methods are overridden by the base class
            // here it does nothing unless overridden
            AddDefaultsToActiveSwitchList();

            AddFallbacksToActiveSwitchList();

            PostProcessSwitchList();

            CommandLineBuilder commandLineBuilder = new CommandLineBuilder(true /* quote hyphens */);

            // iterates through the list of set toolswitches
            foreach (string propertyName in SwitchOrderList)
            {
                if (IsPropertySet(propertyName))
                {
                    ToolSwitch property = activeToolSwitches[propertyName];

                    // verify the dependencies
                    if (VerifyDependenciesArePresent(property) && VerifyRequiredArgumentsArePresent(property, false))
                    {
                        bool addSwitch = true;

                        if (switchesToRemove != null)
                        {
                            foreach (string switchToRemove in switchesToRemove)
                            {
                                if (propertyName.Equals(switchToRemove, StringComparison.OrdinalIgnoreCase))
                                {
                                    addSwitch = false;
                                    break;
                                }
                            }
                        }

                        if (addSwitch && !IsArgument(property))
                        {
                            GenerateCommandsAccordingToType(commandLineBuilder, property, format, escapeFormat);
                        }
                    }
                }
                else if (String.Equals(propertyName, "additionaloptions", StringComparison.OrdinalIgnoreCase))
                {
                    BuildAdditionalArgs(commandLineBuilder);
                    hasSetAdditionalOptions = true;
                }
                else if (String.Equals(propertyName, "AlwaysAppend", StringComparison.OrdinalIgnoreCase))
                {
                    commandLineBuilder.AppendSwitch(AlwaysAppend);
                }
            }

            // if AdditionalOptions hasn't been set, then add additional args to the end
            if (hasSetAdditionalOptions == false)
            {
                BuildAdditionalArgs(commandLineBuilder);
            }

            return commandLineBuilder.ToString();
        }

        /// <summary>
        /// Allows tool to handle the return code.
        /// This method will only be called with non-zero exitCode. If the non zero code is an acceptable one then we return true
        /// </summary>
        /// <returns>The return value of this method will be used as the task return value</returns>
        protected override bool HandleTaskExecutionErrors()
        {
            if (IsAcceptableReturnValue())
                return true;
            // by default, always fail the task
            return base.HandleTaskExecutionErrors();
        }

        #endregion

        #region Public Methods
        /// <summary>
        /// Override Execute so that we can close the event handle we've created
        /// </summary>
        public override bool Execute()
        {
            if (fCancelled)
                return false;

            bool success = base.Execute();

            VCTaskNativeMethods.CloseHandle(cancelEvent);

            // Flush any remaining lines
            PrintMessage(ParseLine(null), StandardOutputImportanceToUse);

            return success;
        }

        /// <summary>
        /// Override Execute so that we can capture the ResolvedPathToTool
        /// </summary>
        protected override int ExecuteTool(
            string pathToTool,
            string responseFileCommands,
            string commandLineCommands)
        {
            ResolvedPathToTool = Environment.ExpandEnvironmentVariables(pathToTool);
            return base.ExecuteTool(pathToTool, responseFileCommands, commandLineCommands);
        }

        public override void Cancel()
        {
            fCancelled = true;
            // We do NOT call the base class version of this because we don't want it trying to kill the task out from under us.
            VCTaskNativeMethods.SetEvent(cancelEvent);
        }

        #endregion // Public Methods

        #region Protected Methods

        /// <summary>
        /// Verifies that the required args are present. This function throws if we have missing required args
        /// </summary>
        /// <param name="property"></param>
        /// <returns></returns>
        protected bool VerifyRequiredArgumentsArePresent(ToolSwitch property, bool throwOnError)
        {
            // check to see if we have required arguments and if there are, that they are present

            if (property.ArgumentRelationList != null)
            {
                foreach (ArgumentRelation relation in property.ArgumentRelationList)
                {
                    if (relation.Required && (property.Value == relation.Value || relation.Value == String.Empty) && !HasSwitch(relation.Argument))
                    {
                        string formattedMessage = "";

                        if (String.Empty == relation.Value)
                            formattedMessage = Log.FormatResourceString("MissingRequiredArgument", relation.Argument, property.Name);
                        else
                            formattedMessage = Log.FormatResourceString("MissingRequiredArgumentWithValue", relation.Argument, property.Name, relation.Value);

                        Log.LogError(formattedMessage);

                        if (throwOnError)
                        {
                            throw new LoggerException(formattedMessage);
                        }

                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Given the ExitCode property, returns true if the return value indicates success, and
        /// false otherwise.
        /// </summary>
        protected bool IsAcceptableReturnValue()
        {
            return IsAcceptableReturnValue(this.ExitCode);
        }

        /// <summary>
        /// Returns true for success code, false otherwise.
        /// </summary>
        /// <param name="code"></param>
        /// <returns></returns>
        protected bool IsAcceptableReturnValue(int code)
        {
            if (AcceptableNonzeroExitCodes != null)
            {
                foreach (string acceptableExitCode in AcceptableNonzeroExitCodes)
                {
                    if (code == Convert.ToInt32(acceptableExitCode, CultureInfo.InvariantCulture))
                        return true;
                }
            }

            return code == 0;
        }
        
        protected void RemoveSwitchToolBasedOnValue(string switchValue)
        {
            if (ActiveToolSwitchesValues.Count > 0)
            {
                if (ActiveToolSwitchesValues.ContainsKey("/" + switchValue))
                {
                    ToolSwitch toolSwitch = ActiveToolSwitchesValues["/" + switchValue];

                    if (toolSwitch != null)
                        ActiveToolSwitches.Remove(toolSwitch.Name);
                }
            }
        }

        protected void AddActiveSwitchToolValue(ToolSwitch switchToAdd)
        {
            if (((switchToAdd.Type != ToolSwitchType.Boolean)
                            || (switchToAdd.BooleanValue == true)))
            {
                if ((switchToAdd.SwitchValue != String.Empty))
                {
                    ActiveToolSwitchesValues.Add(switchToAdd.SwitchValue, switchToAdd);
                }
            }
            else
            {
                if ((switchToAdd.ReverseSwitchValue != String.Empty))
                {
                    ActiveToolSwitchesValues.Add(switchToAdd.ReverseSwitchValue, switchToAdd);
                }
            }
        }

        /// <summary>
        /// Checks to see if the argument is required and whether an argument exists, and returns the
        /// argument or else fallback argument if it exists.
        ///
        /// These are the conditions to look at:
        ///
        /// ArgumentRequired    ArgumentParameter   FallbackArgumentParameter   Result
        /// true                isSet               NA                          The value in ArgumentParameter gets returned
        /// true                isNotSet            isSet                       The value in FallbackArgumentParamter gets returned
        /// true                isNotSet            isNotSet                    An error occurs, as argumentrequired is true
        /// false               isSet               NA                          The value in ArgumentParameter gets returned
        /// false               isNotSet            isSet                       The value in FallbackArgumentParameter gets returned
        /// false               isNotSet            isNotSet                    The empty string is returned, as there are no arguments, and no arguments are required
        /// </summary>
        protected string GetEffectiveArgumentsValues(ToolSwitch property, CommandLineFormat format = CommandLineFormat.ForBuildLog)
        {
            StringBuilder argumentsValues = new StringBuilder();
            bool HasMultipleArguments = false;
            string currentArgumentName = String.Empty;

            if (property.ArgumentRelationList != null)
            {
                foreach (ArgumentRelation relation in property.ArgumentRelationList)
                {
                    if (currentArgumentName != String.Empty && currentArgumentName != relation.Argument)
                        HasMultipleArguments = true;

                    currentArgumentName = relation.Argument;

                    if ((property.Value == relation.Value || relation.Value == String.Empty || (property.Type == ToolSwitchType.Boolean && property.BooleanValue)) && HasSwitch(relation.Argument)) //IsPropertySet(propertyName)
                    {
                        ToolSwitch argument = ActiveToolSwitches[relation.Argument];
                        argumentsValues.Append(relation.Separator);

                        CommandLineBuilder builder1 = new CommandLineBuilder();
                        GenerateCommandsAccordingToType(builder1, argument, format);

                        argumentsValues.Append(builder1.ToString());
                    }
                }
            }

            CommandLineBuilder builder = new CommandLineBuilder();

            if (HasMultipleArguments)
                builder.AppendSwitchIfNotNull("", argumentsValues.ToString());
            else
                builder.AppendSwitchUnquotedIfNotNull("", argumentsValues.ToString());

            return builder.ToString();
        }

        protected virtual void PostProcessSwitchList()
        {
            ValidateRelations();
            ValidateOverrides();
        }

        protected virtual void ValidateRelations()
        {
        }

        protected virtual void ValidateOverrides()
        {
            List<string> overridedSwitches = new List<string>();

            //Collect the overrided switches
            foreach (KeyValuePair<string, ToolSwitch> it in ActiveToolSwitches)
            {
                foreach (KeyValuePair<string, string> overridden in it.Value.Overrides)
                {
                    if (String.Equals(overridden.Key, (it.Value.Type == ToolSwitchType.Boolean && it.Value.BooleanValue == false) ? it.Value.ReverseSwitchValue.TrimStart('/') : it.Value.SwitchValue.TrimStart('/'), StringComparison.OrdinalIgnoreCase))
                    {
                        foreach (KeyValuePair<string, ToolSwitch> itOverridden in ActiveToolSwitches)
                        {
                            if (!String.Equals(itOverridden.Key, it.Key, StringComparison.OrdinalIgnoreCase))
                            {
                                if (String.Equals(itOverridden.Value.SwitchValue.TrimStart('/'), overridden.Value, StringComparison.OrdinalIgnoreCase))
                                {
                                    overridedSwitches.Add(itOverridden.Key);
                                    break;
                                }
                                else if (itOverridden.Value.Type == ToolSwitchType.Boolean && itOverridden.Value.BooleanValue == false && String.Equals(itOverridden.Value.ReverseSwitchValue.TrimStart('/'), overridden.Value, StringComparison.OrdinalIgnoreCase))
                                {
                                    overridedSwitches.Add(itOverridden.Key);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            //Remove the overrided switches
            foreach (string overridenSwitch in overridedSwitches)
            {
                ActiveToolSwitches.Remove(overridenSwitch);
            }
        }

        /// <summary>
        /// Returns true if the property has a value in the list of active tool switches
        /// </summary>
        protected bool IsSwitchValueSet(string switchValue)
        {
            if (!String.IsNullOrEmpty(switchValue))
            {
                return ActiveToolSwitchesValues.ContainsKey("/" + switchValue);
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Verifies that the dependencies are present, and if the dependencies are present, or if the property
        /// doesn't have any dependencies, the switch gets emitted
        /// </summary>
        /// <param name="property"></param>
        /// <returns></returns>
        protected virtual bool VerifyDependenciesArePresent(ToolSwitch value)
        {
            // check the dependency
            if (value.Parents.Count > 0)
            {
                // has a dependency, now check to see whether at least one parent is set
                // if it is set, add to the command line
                // otherwise, ignore it
                bool isSet = false;

                foreach (string parentName in value.Parents)
                {
                    isSet = isSet || HasSwitch(parentName);
                }
                return isSet;
            }
            else
            {
                // no dependencies to account for
                return true;
            }
        }

        /// <summary>
        /// A protected method to add the switches that are by default visible
        /// e.g., /nologo is true by default
        /// </summary>
        protected virtual void AddDefaultsToActiveSwitchList()
        {
            // do nothing
        }

        /// <summary>
        /// A method that will add the fallbacks to the active switch list if the actual property is not set
        /// </summary>
        protected virtual void AddFallbacksToActiveSwitchList()
        {
            // do nothing
        }

        /// <summary>
        /// Have to keep for backward compatibility in Dev16. Can be removed in Dev17.
        /// </summary>
        protected virtual void GenerateCommandsAccordingToType(CommandLineBuilder builder, ToolSwitch toolSwitch, bool dummyForBackwardCompatibility, CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.Default)
        {
            GenerateCommandsAccordingToType(builder, toolSwitch, format, escapeFormat);
        }       

        /// <summary>
        /// Generates a part of the command line depending on the type
        /// </summary>
        /// <remarks>Depending on the type of the switch, the switch is emitted with the proper values appended.
        /// e.g., File switches will append file names, directory switches will append filenames with "\" on the end</remarks>
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        protected virtual void GenerateCommandsAccordingToType(CommandLineBuilder builder, ToolSwitch toolSwitch, CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.Default)
        {
            switch (toolSwitch.Type)
            {
                case ToolSwitchType.Boolean:
                    EmitBooleanSwitch(builder, toolSwitch, format);
                    break;
                case ToolSwitchType.String:
                    EmitStringSwitch(builder, toolSwitch);
                    break;
                case ToolSwitchType.StringArray:
                    EmitStringArraySwitch(builder, toolSwitch);
                    break;
                case ToolSwitchType.StringPathArray:
                    EmitStringArraySwitch(builder, toolSwitch, format, escapeFormat);
                    break;
                case ToolSwitchType.Integer:
                    EmitIntegerSwitch(builder, toolSwitch);
                    break;
                case ToolSwitchType.File:
                    EmitFileSwitch(builder, toolSwitch, format);
                    break;
                case ToolSwitchType.Directory:
                    EmitDirectorySwitch(builder, toolSwitch, format);
                    break;
                case ToolSwitchType.ITaskItem:
                    EmitTaskItemSwitch(builder, toolSwitch);
                    break;
                case ToolSwitchType.ITaskItemArray:
                    EmitTaskItemArraySwitch(builder, toolSwitch, format);
                    break;
                case ToolSwitchType.AlwaysAppend:
                    EmitAlwaysAppendSwitch(builder, toolSwitch);
                    break;
                default:
                    // should never reach this point - if it does, there's a bug somewhere.
                    ErrorUtilities.VerifyThrow(false, "InternalError");
                    break;
            }
        }

        /// <summary>
        /// Appends a literal string containing the verbatim contents of any
        /// "AdditionalOptions" parameter. This goes last on the command
        /// line in case it needs to cancel any earlier switch.
        /// Ideally this should never be needed because the MSBuild task model
        /// is to set properties, not raw switches
        /// </summary>
        /// <param name="cmdLine"></param>
        protected void BuildAdditionalArgs(CommandLineBuilder cmdLine)
        {
            // We want additional options to be last so that this can always override other flags.
            if ((cmdLine != null) && !String.IsNullOrEmpty(additionalOptions))
            {
                cmdLine.AppendSwitch(System.Environment.ExpandEnvironmentVariables(additionalOptions));
            }
        }

        /// <summary>
        /// A method that will validate the integer type arguments
        /// If the min or max is set, and the value a property is set to is not within
        /// the range, the build fails
        /// </summary>
        /// <param name="switchName"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="value"></param>
        /// <returns>The valid integer passed converted to a string form</returns>
        protected bool ValidateInteger(string switchName, int min, int max, int value)
        {
            if (value < min || value > max)
            {
                logPrivate.LogErrorFromResources("ArgumentOutOfRange", switchName, value);
                return false;
            }

            return true;
        }
        protected bool IgnoreUnknownSwitchValues { get; set; }

        /// <summary>
        /// A method for the enumerated values a property can have
        /// This method checks the value a property is set to, and finds the corresponding switch
        /// </summary>
        /// <param name="propertyName"></param>
        /// <param name="switchMap"></param>
        /// <param name="value"></param>
        /// <returns>The switch that a certain value is mapped to</returns>
        protected string ReadSwitchMap(string propertyName, string[][] switchMap, string value)
        {
            if (switchMap != null)
            {
                for (int i = 0; i < switchMap.Length; ++i)
                {
                    if (String.Equals(switchMap[i][0], value, StringComparison.CurrentCultureIgnoreCase))
                    {
                        return switchMap[i][1];
                    }
                }
                if (!IgnoreUnknownSwitchValues)
                {
                    logPrivate.LogErrorFromResources("ArgumentOutOfRange", propertyName, value);
                }
            }
            return String.Empty;
        }

        /// <summary>
        /// Returns true if the property has a value in the list of active tool switches
        /// </summary>
        protected bool IsPropertySet(string propertyName)
        {
            return activeToolSwitches.ContainsKey(propertyName);
        }

        /// <summary>
        /// Returns true if the property is set to true.
        /// Returns false if the property is not set, or set to false.
        /// </summary>
        protected bool IsSetToTrue(string propertyName)
        {
            if (activeToolSwitches.ContainsKey(propertyName))
            {
                return activeToolSwitches[propertyName].BooleanValue;
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// Returns true if the property is set to false.
        /// Returns false if the property is not set, or set to true.
        /// </summary>
        protected bool IsExplicitlySetToFalse(string propertyName)
        {
            if (activeToolSwitches.ContainsKey(propertyName))
            {
                return !activeToolSwitches[propertyName].BooleanValue;
            }
            else
            {
                return false;
            }
        }

        protected bool IsArgument(ToolSwitch property)
        {
            if(property != null && property.Parents.Count > 0)
            {
                if (String.IsNullOrEmpty(property.SwitchValue))
                    return true;

                foreach (var parentName in property.Parents)
                {
                    ToolSwitch parentSwitch;
                    if (activeToolSwitches.TryGetValue(parentName, out parentSwitch))
                    {
                        foreach (ArgumentRelation argumentRelation in parentSwitch.ArgumentRelationList)
                        {
                            if (argumentRelation.Argument.Equals(property.Name, StringComparison.Ordinal))
                            {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Checks to see if the switch name is empty
        /// </summary>
        /// <param name="propertyName"></param>
        /// <returns></returns>
        protected bool HasSwitch(string propertyName)
        {
            if (IsPropertySet(propertyName))
            {
                return !String.IsNullOrEmpty(activeToolSwitches[propertyName].Name);
            }
            else
            {
                return false;
            }
        }

        /// <summary>
        /// If the given path doesn't have a trailing slash then add one.
        /// </summary>
        /// <param name="directoryName">The path to check.</param>
        /// <returns>A path with a slash.</returns>
        protected static string EnsureTrailingSlash(string directoryName)
        {
            ErrorUtilities.VerifyThrow(directoryName != null, "InternalError");
            if (directoryName is object && directoryName.Length > 0)
            {
                char endingCharacter = directoryName[directoryName.Length - 1];

                if (!(endingCharacter == Path.DirectorySeparatorChar
                    || endingCharacter == Path.AltDirectorySeparatorChar))
                {
                    directoryName += Path.DirectorySeparatorChar;
                }
            }

            return directoryName;
        }

        protected List<Regex> errorListRegexList = new List<Regex>();
        protected List<Regex> errorListRegexListExclusion = new List<Regex>();
        protected MessageStruct lastMS = new MessageStruct();
        protected MessageStruct currentMS = new MessageStruct();

        /// <summary>
        /// Override behavior to handle multiline messages
        /// </summary>
        protected override void LogEventsFromTextOutput(string singleLine, MessageImportance messageImportance)
        {
            if (EnableErrorListRegex && errorListRegexList.Count > 0)
            {
                PrintMessage(ParseLine(singleLine), messageImportance);

                if (ForceSinglelineErrorListRegex)
                    PrintMessage(ParseLine(null), messageImportance);
            }
            else
            {
                base.LogEventsFromTextOutput(singleLine, messageImportance);
            }
        }

        protected virtual void PrintMessage(MessageStruct message, MessageImportance messageImportance)
        {
            if (message != null && message.Text.Length > 0)
            {
                switch (message.Category)
                {
                    case "fatal error":
                    case "error":
                        {
                            Log.LogError(message.SubCategory, message.Code, null, message.Filename, message.Line, message.Column, 0, 0, message.Text.TrimEnd());
                            break;
                        }
                    case "warning":
                        {
                            Log.LogWarning(message.SubCategory, message.Code, null, message.Filename, message.Line, message.Column, 0, 0, message.Text.TrimEnd());
                            break;
                        }
                    case "note":
                        {
                            Log.LogCriticalMessage(message.SubCategory, message.Code, null, message.Filename, message.Line, message.Column, 0, 0, message.Text.TrimEnd());
                            break;
                        }
                    default:
                        {
                            Log.LogMessage(messageImportance, message.Text.TrimEnd());
                            break;
                        }
                }

                message.Clear();
            }
        }

        /// <summary>
        /// Parse the inputs, once a multi-line block is done, then returns the current message struct.
        /// </summary>
        /// <remarks>When input is null, then this will immediately return the current message.</remarks>
        protected virtual MessageStruct ParseLine(string inputLine)
        {
            if (inputLine == null)
            {
                // return and stop collecting until next match line
                MessageStruct.Swap(ref lastMS, ref currentMS);
                currentMS.Clear();
                return lastMS;
            }

            if (String.IsNullOrWhiteSpace(inputLine))
            {
                return null;
            }

            bool matchSuccess = false;

            foreach (var regex in errorListRegexListExclusion)
            {
                try
                {
                    var match = regex.Match(inputLine);

                    if (match.Success)
                    {
                        matchSuccess = true;
                        break;
                    }
                }
                catch (RegexMatchTimeoutException)
                { // treat as no match
                }
                catch (Exception e)
                {
                    ReportRegexException(inputLine, regex, e);
                }
            }

            if (!matchSuccess)
            {
                foreach (var regex in errorListRegexList)
                {
                    try
                    {
                        var match = regex.Match(inputLine);

                        if (match.Success)
                        {
                            int line = 0, column = 0;

                            if (!Int32.TryParse(match.Groups["LINE"].Value, out line)) line = 0;
                            if (!Int32.TryParse(match.Groups["COLUMN"].Value, out column)) column = 0;

                            MessageStruct.Swap(ref lastMS, ref currentMS);

                            currentMS.Clear();
                            currentMS.Category = match.Groups["CATEGORY"].Value.ToLowerInvariant();
                            currentMS.SubCategory = match.Groups["SUBCATEGORY"].Value.ToLowerInvariant();
                            currentMS.Filename = match.Groups["FILENAME"].Value;
                            currentMS.Code = match.Groups["CODE"].Value;
                            currentMS.Line = line;
                            currentMS.Column = column;
                            currentMS.Text += match.Groups["TEXT"].Value.TrimEnd() + Environment.NewLine;
                            matchSuccess = true;
                            return lastMS;
                        }
                    }
                    catch (RegexMatchTimeoutException)
                    { // treat as no match
                    }
                    catch (Exception e)
                    {
                        ReportRegexException(inputLine, regex, e);
                    }
                }
            }

            if (!matchSuccess && !String.IsNullOrEmpty(currentMS.Filename))
            {
                // append line to current messabe block
                currentMS.Text += inputLine.TrimEnd() + Environment.NewLine;
                return null;
            }
            else
            {
                // the message block has been exclusioned, so print the current message
                MessageStruct.Swap(ref lastMS, ref currentMS);
                currentMS.Clear();
                currentMS.Text = inputLine;
                return lastMS;
            }
        }

        protected void ReportRegexException(string inputLine, Regex regex, Exception e)
        {
            if (ExceptionHandling.IsCriticalException(e))
            {
                if (e is OutOfMemoryException)
                {
                    // try to free some memory before logging
                    int CurCacheSize = Regex.CacheSize;
                    Regex.CacheSize = 0;
                    Regex.CacheSize = CurCacheSize;
                }

                this.Log.LogErrorWithCodeFromResources("TrackedVCToolTask.CannotParseToolOutput", inputLine, regex.ToString(), e.Message);
                e.Rethrow();
            }
            else
            {
                this.Log.LogWarningWithCodeFromResources("TrackedVCToolTask.CannotParseToolOutput", inputLine, regex.ToString(), e.Message);
            }    
        }

        protected class MessageStruct
        {
            // Fatal error, Error, warning, or note
            public string Category { get; set; } = "";

            // Fatal, Command line
            public string SubCategory { get; set; } = "";

            // If error has a code for easy online look up
            public string Code { get; set; } = "";

            // Goto file/line/Column of the issue
            public string Filename { get; set; } = "";

            public int Line { get; set; }

            public int Column { get; set; }

            // The description of the error
            public string Text { get; set; } = "";

            public void Clear()
            {
                this.Category = "";
                this.SubCategory = "";
                this.Code = "";
                this.Filename = "";
                this.Line = 0;
                this.Column = 0;
                this.Text = "";
            }

            public static void Swap(ref MessageStruct lhs, ref MessageStruct rhs)
            {
                MessageStruct temp = lhs;
                lhs = rhs;
                rhs = temp;
            }
        }

        #endregion // Protected Methods

        #region Private Methods

        /// <summary>
        /// Emit a switch that's always appended
        /// </summary>
        private static void EmitAlwaysAppendSwitch(CommandLineBuilder builder, ToolSwitch toolSwitch)
        {
            builder.AppendSwitch(toolSwitch.Name);
        }

        /// <summary>
        /// Emit a switch that's an array of task items
        /// </summary>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        private static void EmitTaskItemArraySwitch(CommandLineBuilder builder, ToolSwitch toolSwitch, CommandLineFormat format = CommandLineFormat.ForBuildLog)
        {
            if (String.IsNullOrEmpty(toolSwitch.Separator))
            {
                foreach (ITaskItem itemName in toolSwitch.TaskItemArray)
                {
                    builder.AppendSwitchIfNotNull(toolSwitch.SwitchValue, System.Environment.ExpandEnvironmentVariables(itemName.ItemSpec));
                }
            }
            else
            {
                // Need to create a new ITaskITem array so that we can call ToUpperInvariant if this output is for a
                // response file
                ITaskItem[] itemTaskArray = new ITaskItem[toolSwitch.TaskItemArray.Length];

                for (int i = 0; i < toolSwitch.TaskItemArray.Length; i++)
                {
                    itemTaskArray[i] = new TaskItem(System.Environment.ExpandEnvironmentVariables(toolSwitch.TaskItemArray[i].ItemSpec));

                    if (format == CommandLineFormat.ForTracking)
                    {
                        itemTaskArray[i].ItemSpec = itemTaskArray[i].ItemSpec.ToUpperInvariant();
                    }
                }

                // AppendSwitchIfNotNull has an overload for String[], but it handle quotes differently than ItemTask.
                builder.AppendSwitchIfNotNull(toolSwitch.SwitchValue, itemTaskArray, toolSwitch.Separator);
            }
        }

        /// <summary>
        /// Emit a switch that's a scalar task item
        /// </summary>
        private static void EmitTaskItemSwitch(CommandLineBuilder builder, ToolSwitch toolSwitch)
        {
            if (!String.IsNullOrEmpty(toolSwitch.TaskItem.ItemSpec))
            {
                builder.AppendFileNameIfNotNull(System.Environment.ExpandEnvironmentVariables(toolSwitch.TaskItem.ItemSpec + toolSwitch.Separator));
            }
        }

        /// <summary>
        /// Appends the directory name to the end of a switch
        /// Ensure the name ends with a slash
        /// </summary>
        /// <remarks>For directory switches (e.g., TrackerLogDirectory), the toolSwitchName (if it exists) is emitted
        /// along with the FileName which is ensured to have a trailing slash</remarks>
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        private static void EmitDirectorySwitch(CommandLineBuilder builder, ToolSwitch toolSwitch, CommandLineFormat format = CommandLineFormat.ForBuildLog)
        {
            if (!String.IsNullOrEmpty(toolSwitch.SwitchValue))
            {
                if (format == CommandLineFormat.ForBuildLog)
                {
                    builder.AppendSwitch(toolSwitch.SwitchValue + toolSwitch.Separator);
                }
                else
                {
                    builder.AppendSwitch(toolSwitch.SwitchValue.ToUpperInvariant() + toolSwitch.Separator);
                }
            }
        }

        /// <summary>
        /// Generates the switches that have filenames attached to the end
        /// </summary>
        /// <remarks>For file switches (e.g., PrecompiledHeaderFile), the toolSwitchName (if it exists) is emitted
        /// along with the FileName which may or may not have quotes</remarks>
        /// e.g., PrecompiledHeaderFile = "File" will emit /FpFile
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        /// <param name="format">Specifies if the resulting commandline is intended for build output/log or for tracking</param>
        private static void EmitFileSwitch(CommandLineBuilder builder, ToolSwitch toolSwitch, CommandLineFormat format = CommandLineFormat.ForBuildLog)
        {
            if (!String.IsNullOrEmpty(toolSwitch.Value))
            {
                String str = System.Environment.ExpandEnvironmentVariables(toolSwitch.Value);
                str = str.Trim();

                if (format == CommandLineFormat.ForTracking)
                {
                    str = str.ToUpperInvariant();
                }

                if (!str.StartsWith("\"", StringComparison.Ordinal))
                {
                    str = "\"" + str;
                    if (str.EndsWith("\\", StringComparison.Ordinal) && !str.EndsWith("\\\\", StringComparison.Ordinal))
                        str += "\\\"";
                    else
                        str += "\"";
                }

                //we want quotes always, AppendSwitchIfNotNull will add them on as needed bases
                builder.AppendSwitchUnquotedIfNotNull(toolSwitch.SwitchValue + toolSwitch.Separator, str);
            }
        }

        /// <summary>
        /// Generates the commands for switches that have integers appended.
        /// </summary>
        /// <remarks>For integer switches (e.g., WarningLevel), the toolSwitchName is emitted
        /// with the appropriate integer appended, as well as any arguments
        /// e.g., WarningLevel = "4" will emit /W4</remarks>
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        private void EmitIntegerSwitch(CommandLineBuilder builder, ToolSwitch toolSwitch)
        {
            if (toolSwitch.IsValid)
            {
                if (!String.IsNullOrEmpty(toolSwitch.Separator))
                {
                    builder.AppendSwitch(toolSwitch.SwitchValue + toolSwitch.Separator + toolSwitch.Number.ToString(CultureInfo.InvariantCulture) + GetEffectiveArgumentsValues(toolSwitch));
                }
                else
                {
                    builder.AppendSwitch(toolSwitch.SwitchValue + toolSwitch.Number.ToString(CultureInfo.InvariantCulture) + GetEffectiveArgumentsValues(toolSwitch));
                }
            }
        }

        /// <summary>
        /// Generates the commands for the switches that may have an array of arguments
        /// The switch may be empty.
        /// </summary>
        /// <remarks>For stringarray switches (e.g., Sources), the toolSwitchName (if it exists) is emitted
        /// along with each and every one of the file names separately (if no separator is included), or with all of the
        /// file names separated by the separator.
        /// e.g., AdditionalIncludeDirectores = "@(Files)" where Files has File1, File2, and File3, the switch
        /// /IFile1 /IFile2 /IFile3 or the switch /IFile1;File2;File3 is emitted (the latter case has a separator
        /// ";" specified)</remarks>
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        private static void EmitStringArraySwitch(CommandLineBuilder builder, ToolSwitch toolSwitch, CommandLineFormat format = CommandLineFormat.ForBuildLog, EscapeFormat escapeFormat = EscapeFormat.Default)
        {
            string[] ArrTrimStringList = new string[toolSwitch.StringList.Length];
            char[] requireQuotes = { ' ', '|', '<', '>', ',', ';', '-', '\r', '\n', '\t', '\f', };

            for (int i = 0; i < toolSwitch.StringList.Length; ++i)
            {
                string buffer;
                //Make sure the file doesn't contain escaped " (\")
                if (toolSwitch.StringList[i].StartsWith("\"", StringComparison.Ordinal) && toolSwitch.StringList[i].EndsWith("\"", StringComparison.Ordinal))
                {
                    buffer = System.Environment.ExpandEnvironmentVariables(toolSwitch.StringList[i].Substring(1, toolSwitch.StringList[i].Length - 2));
                }
                else
                {
                    buffer = System.Environment.ExpandEnvironmentVariables(toolSwitch.StringList[i]);
                }

                if (!String.IsNullOrEmpty(buffer))
                {
                    if (format == CommandLineFormat.ForTracking)
                    {
                        buffer = buffer.ToUpperInvariant();
                    }

                    if (escapeFormat.HasFlag(EscapeFormat.EscapeTrailingSlash) && buffer.IndexOfAny(requireQuotes) == -1 && buffer.EndsWith("\\", StringComparison.Ordinal) && !buffer.EndsWith("\\\\", StringComparison.Ordinal))
                    {
                        // if the value ends with a slash, then the slash could escape the following space
                        // unless the value contains a space, then it AppendSwitchIfNotNull() add quotes
                        buffer += "\\";
                    }

                    ArrTrimStringList[i] = buffer;
                }
            }

            if (String.IsNullOrEmpty(toolSwitch.Separator))
            {
                foreach (string fileName in ArrTrimStringList)
                {
                    builder.AppendSwitchIfNotNull(toolSwitch.SwitchValue, fileName);
                }
            }
            else
            {
                builder.AppendSwitchIfNotNull(toolSwitch.SwitchValue, ArrTrimStringList, toolSwitch.Separator);
            }
        }

        /// <summary>
        /// Generates the switches for switches that either have literal strings appended, or have
        /// different switches based on what the property is set to.
        /// </summary>
        /// <remarks>The string switch emits a switch that depends on what the parameter is set to, with and
        /// arguments
        /// e.g., Optimization = "Full" will emit /Ox, whereas Optimization = "Disabled" will emit /Od</remarks>
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        private void EmitStringSwitch(CommandLineBuilder builder, ToolSwitch toolSwitch)
        {
            String strSwitch = String.Empty;
            strSwitch += toolSwitch.SwitchValue + toolSwitch.Separator;

            StringBuilder val = new StringBuilder(GetEffectiveArgumentsValues(toolSwitch));
            String str = toolSwitch.Value;

            if (!toolSwitch.MultipleValues)
            {
                str = str.Trim();

                if (!str.StartsWith("\"", StringComparison.Ordinal))
                {
                    str = "\"" + str;
                    if (str.EndsWith("\\", StringComparison.Ordinal) && !str.EndsWith("\\\\", StringComparison.Ordinal))
                        str += "\\\"";
                    else
                        str += "\"";
                }

                val.Insert(0, str);
            }

            if ((strSwitch.Length == 0) && (val.ToString().Length == 0))
                return;

            builder.AppendSwitchUnquotedIfNotNull(strSwitch, val.ToString());
        }

        /// <summary>
        /// Generates the switches that are nonreversible
        /// </summary>
        /// <remarks>A boolean switch is emitted if it is set to true. If it set to false, nothing is emitted.
        /// e.g. nologo = "true" will emit /Og, but nologo = "false" will emit nothing.</remarks>
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        private void EmitBooleanSwitch(CommandLineBuilder builder, ToolSwitch toolSwitch, CommandLineFormat format = CommandLineFormat.ForBuildLog)
        {
            if (toolSwitch.BooleanValue)
            {
                if (!String.IsNullOrEmpty(toolSwitch.SwitchValue))
                {
                    StringBuilder val = new StringBuilder(GetEffectiveArgumentsValues(toolSwitch, format));
                    val.Insert(0, toolSwitch.Separator);
                    val.Insert(0, toolSwitch.TrueSuffix);
                    val.Insert(0, toolSwitch.SwitchValue);
                    builder.AppendSwitch(val.ToString());
                }
            }
            else
                EmitReversibleBooleanSwitch(builder, toolSwitch);
        }

        /// <summary>
        /// Generates the command line for switches that are reversible
        /// </summary>
        /// <remarks>A reversible boolean switch will emit a certain switch if set to true, but emit that
        /// exact same switch with a flag appended on the end if set to false.
        /// e.g., GlobalOptimizations = "true" will emit /Og, and GlobalOptimizations = "false" will emit /Og-</remarks>
        /// <param name="builder"></param>
        /// <param name="toolSwitch"></param>
        private void EmitReversibleBooleanSwitch(CommandLineBuilder builder, ToolSwitch toolSwitch)
        {
            // if the value is set to true, append whatever the TrueSuffix is set to.
            // Otherwise, append whatever the FalseSuffix is set to.
            if (!String.IsNullOrEmpty(toolSwitch.ReverseSwitchValue))
            {
                string suffix = (toolSwitch.BooleanValue) ? toolSwitch.TrueSuffix : toolSwitch.FalseSuffix;
                StringBuilder val = new StringBuilder(GetEffectiveArgumentsValues(toolSwitch));
                val.Insert(0, suffix);
                val.Insert(0, toolSwitch.Separator);
                val.Insert(0, toolSwitch.TrueSuffix);
                val.Insert(0, toolSwitch.ReverseSwitchValue);
                builder.AppendSwitch(val.ToString());
            }
        }

        /// <summary>
        /// Checks to make sure that a switch has either a '/' or a '-' prefixed.
        /// </summary>
        /// <param name="toolSwitch"></param>
        /// <returns></returns>
        private string Prefix(string toolSwitch)
        {
            if (!String.IsNullOrEmpty(toolSwitch))
            {
                if (toolSwitch[0] != prefix)
                {
                    return prefix + toolSwitch;
                }
            }
            return toolSwitch;
        }

        #endregion Private Methods
    }

    /// <summary>
    /// Private class wrapping the PInvoke calls required by VCToolTask
    /// </summary>
    internal static class VCTaskNativeMethods
    {
        [DllImport("KERNEL32.DLL", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern IntPtr CreateEventW(IntPtr lpEventAttributes, bool bManualReset, bool bInitialState, string lpName);

        [DllImport("KERNEL32.DLL", SetLastError = true)]
        internal static extern bool SetEvent(IntPtr hEvent);

        [DllImport("KERNEL32.DLL", SetLastError = true)]
        internal static extern bool CloseHandle(IntPtr hObject);

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        internal static extern uint SearchPath(string lpPath,
                         string lpFileName,
                         string lpExtension,
                         int nBufferLength,
                         [MarshalAs(UnmanagedType.LPTStr)]
                             StringBuilder lpBuffer,
                         out IntPtr lpFilePart);
    }
}