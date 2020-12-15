// -----------------------------------------------------------------------
// <copyright file="ClCmdArgsCaptureTool.cs" company="Microsoft Corporation">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root
// for license information.
// </copyright>
// -----------------------------------------------------------------------

namespace CodeBook
{
    using System;

    /// <summary>
    /// The ClCmdArgsCaptureTool simply logs the arguments its been passed to a SQLite database.
    /// </summary>
    /// <remarks>
    /// See README.md for additional details.
    /// </remarks>
    public static class ClCmdArgsCaptureTool
    {
        /// <summary>
        /// The main function for the ClCmdArgsCaptureTool.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        /// <returns>0 on success (always), else non-zero.</returns>
        public static int Main(string[] args)
        {
            if (args == null)
            {
                args = Array.Empty<string>();
            }

            foreach (string arg in args)
            {
                System.Console.WriteLine(arg);
            }

            // Always report success.
            return 0;
        }
    }
}