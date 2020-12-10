using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Json;
using System.Text;
using Microsoft.Build.Framework;

namespace Microsoft.Build.Tasks
{
    /// <summary>
    /// An msbuild task to output a clang style json compilation database.
    /// See Also: https://clang.llvm.org/docs/JSONCompilationDatabase.html
    /// </summary>
    public class GenerateCompilationDatabaseTask : Task
    {
        public CompilationDatabaseTask()
        {
        }

        /// <summary>
        /// Property name: BuildPath
        /// Property type: string
        /// Switch: p=
        /// </summary>
        public virtual string BuildPath
        {
            get
            {
                if (this.IsPropertySet("BuildPath"))
                {
                    return ActiveToolSwitches["BuildPath"].Value;
                }
                else
                {
                    return string.Empty;
                }
            }
            set
            {
                this.ActiveToolSwitches.Remove("BuildPath");
                ToolSwitch switchToAdd = new ToolSwitch(ToolSwitchType.String)
                {
                    DisplayName = "Build path",
                    Description = "Build path to search for compile commands database.",
                    ArgumentRelationList = new ArrayList(),
                    SwitchValue = "-p=",
                    Name = "BuildPath",
                    Value = value
                };
                this.ActiveToolSwitches.Add("BuildPath", switchToAdd);
                this.AddActiveSwitchToolValue(switchToAdd);
            }
        }

        /// <summary>
        /// Property name: Checks
        /// Property type: String
        /// Switch: checks=
        /// </summary>
        public virtual string Checks
        {
            get
            {
                if (this.IsPropertySet("Checks"))
                {
                    return ActiveToolSwitches["Checks"].Value;
                }
                else
                {
                    return null;
                }
            }
            set
            {
                this.ActiveToolSwitches.Remove("Checks");
                ToolSwitch switchToAdd = new ToolSwitch(ToolSwitchType.String)
                {
                    DisplayName = "Checks",
                    Description = "Comma-separated list of globs denoting which checks to enable/disable (use prefix '-' to disable)",
                    ArgumentRelationList = new ArrayList(),
                    Name = "Checks",
                    Value = value,
                    SwitchValue = "--checks="
                };
                this.ActiveToolSwitches.Add("Checks", switchToAdd);
                this.AddActiveSwitchToolValue(switchToAdd);
            }
        }

        /// <summary>
        /// Property name: HeaderFilter
        /// Property type: string
        /// Switch: -header-filter
        /// </summary>
        public virtual bool HeaderFilter
        {
            get
            {
                if (this.IsPropertySet("HeaderFilter"))
                {
                    return ActiveToolSwitches["HeaderFilter"].BooleanValue;
                }
                else
                {
                    return false;
                }
            }
            set
            {
                this.ActiveToolSwitches.Remove("HeaderFilter");
                ToolSwitch switchToAdd = new ToolSwitch(ToolSwitchType.String)
                {
                    DisplayName = "HeaderFilter",
                    Description = "Regular expression matching the names of the headers to output diagnostics from.",
                    ArgumentRelationList = new ArrayList(),
                    Name = "HeaderFilter",
                    BooleanValue = value,
                    SwitchValue = "--header-filter="
                };
                this.ActiveToolSwitches.Add("HeaderFilter", switchToAdd);
                this.AddActiveSwitchToolValue(switchToAdd);
            }
        }

        /// <summary>
        /// Property name: WarningsInSystemHeaders
        /// Property type: boolean
        /// Switch: -system-headers
        /// </summary>
        public virtual bool WarningsInSystemHeaders
        {
            get
            {
                if (this.IsPropertySet("WarningsInSystemHeaders"))
                {
                    return ActiveToolSwitches["WarningsInSystemHeaders"].BooleanValue;
                }
                else
                {
                    return false;
                }
            }
            set
            {
                this.ActiveToolSwitches.Remove("WarningsInSystemHeaders");
                ToolSwitch switchToAdd = new ToolSwitch(ToolSwitchType.Boolean)
                {
                    DisplayName = "WarningsInSystemHeaders",
                    Description = "Display warnings and errors from system headers.",
                    ArgumentRelationList = new ArrayList(),
                    Name = "WarningsInSystemHeaders",
                    BooleanValue = value,
                    SwitchValue = "--system-headers"
                };
                this.ActiveToolSwitches.Add("WarningsInSystemHeaders", switchToAdd);
                this.AddActiveSwitchToolValue(switchToAdd);
            }
        }

        /// <summary>
        /// Property name: CompileCommands
        /// Property type: ITaskItemArray
        /// Switch: 
        /// </summary>
        [Required()]
        public virtual ITaskItem[] CompileCommands { get; set; }

        /// <summary>
        /// Property name: OutputFile
        /// Property type: string
        /// </summary>
        public virtual string OutputFile { get; set; }

        /// <summary>
        /// Override ExecuteTool to print out the compiling files
        /// </summary>
        protected override int ExecuteTool(string pathToTool, string responseFileCommands, string commandLineCommands)
        {
            if (!string.IsNullOrEmpty(OutputFile))
            {
                if (File.Exists(OutputFile))
                    File.Delete(OutputFile);

                using (File.Create(OutputFile)) { }
            }

            // Write compile_commands.json database clang uses to determine compiler commands
            Directory.CreateDirectory(BuildPath);
            var compileCommandFile = Path.Combine(BuildPath, "compile_commands.json");
            File.WriteAllText(compileCommandFile, ConvertJson(CompileCommands));

            var clangTidyCommandArgs = new StringBuilder();

            // Add source files
            clangTidyCommandArgs.Append(string.Join(" ", CompileCommands
                .SelectMany(GetFiles)
                .Select(x => $"\"{x}\"")));

            clangTidyCommandArgs.Append($" {responseFileCommands}");
            clangTidyCommandArgs.Append($" {commandLineCommands}");
            var clangTidyCommandArgsString = clangTidyCommandArgs.ToString();

            // It's not documented, but Clang-Tidy supports response files.
            // Use them if the command is long enough to trigger an error.
            if (clangTidyCommandArgsString.Length > 32000)
            {
                return base.ExecuteTool(pathToTool, clangTidyCommandArgs.ToString(), string.Empty);
            }
            else
            {
                return base.ExecuteTool(pathToTool, string.Empty, clangTidyCommandArgs.ToString());
            }
        }

        protected override void LogEventsFromTextOutput(string singleLine, MessageImportance messageImportance)
        {
            if (!string.IsNullOrEmpty(OutputFile) && !string.IsNullOrEmpty(singleLine))
                File.AppendAllText(OutputFile, singleLine + "\n");

            base.LogEventsFromTextOutput(singleLine, messageImportance);
        }

        private static string ConvertJson(IEnumerable<ITaskItem> compileCommands)
        {
            using (var stream = new MemoryStream())
            {
                using (var jsonWriter = JsonReaderWriterFactory.CreateJsonWriter(stream, Encoding.UTF8, true, true, "  "))
                {
                    jsonWriter.WriteStartElement("root");
                    jsonWriter.WriteAttributeString("type", "array");
                    foreach (var command in compileCommands)
                    {
                        var commandLine = $"\"{command.GetMetadata("ToolPath")}\" {command.ItemSpec}";
                        foreach (var file in GetFiles(command))
                        {
                            jsonWriter.WriteStartElement("item");
                            jsonWriter.WriteAttributeString("type", "object");
                            jsonWriter.WriteElementString("directory", command.GetMetadata("WorkingDirectory"));
                            jsonWriter.WriteElementString("command", $"{commandLine} \"{file}\"");
                            jsonWriter.WriteElementString("file", file);
                            jsonWriter.WriteEndElement();
                        }
                    }

                    jsonWriter.WriteEndElement();
                    jsonWriter.Flush();
                    return Encoding.UTF8.GetString(stream.ToArray());
                }
            }
        }

        private static IEnumerable<string> GetFiles(ITaskItem commandItem)
        {
            return commandItem.GetMetadata("Files")
                .Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
        }
    }
}

