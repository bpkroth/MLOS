// -----------------------------------------------------------------------
// <copyright file="GenerateCompilationDatabaseTask.cs" company="Microsoft Corporation">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root
// for license information.
// </copyright>
// -----------------------------------------------------------------------

using System;

using Microsoft.Build.CPPTasks;

namespace CodeBook.MSBuild.Extensions
{
    /// <summary>
    /// An msbuild task to output a clang style json compilation database.
    /// See Also: https://clang.llvm.org/docs/JSONCompilationDatabase.html.
    /// </summary>
    public class GenerateCompilationDatabaseTask : Microsoft.Build.CPPTasks.ClangCompile
    {
        /// <summary>
        /// Override ExecuteTool to print out the compiling files
        /// </summary>
        protected override int ExecuteTool(string pathToTool, string responseFileCommands, string commandLineCommands)
        {

        }
    }
}
