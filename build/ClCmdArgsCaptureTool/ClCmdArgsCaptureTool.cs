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
    using System.Collections.Generic;
    using System.Data;
    using System.IO;
    using System.Linq;
    using Microsoft.Data.Sqlite;

    /// <summary>
    /// The ClCmdArgsCaptureTool simply logs the arguments its been passed to a SQLite database.
    /// </summary>
    /// <remarks>
    /// See README.md for additional details.
    /// </remarks>
    public static class ClCmdArgsCaptureTool
    {
        /// <summary>
        /// The name of an environment variable that callers are expected to have populated with a
        /// path to the output sqlite db.
        /// </summary>
        private const string SqliteDbFileEnvVarName = "ClCmdArgsCaptureToolOutputDbFile";

        /// <summary>
        /// The main function for the ClCmdArgsCaptureTool.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        /// <returns>0 on success (always), else non-zero.</returns>
        public static int Main(string[] args)
        {
            // Check for null to make stylecop happy.
            //
            if (args == null)
            {
                args = Array.Empty<string>();
            }

            // Determine the output database file from an environment variable.
            //
            string sqliteDbFilePath = System.Environment.GetEnvironmentVariable(SqliteDbFileEnvVarName);
            if (string.IsNullOrEmpty(sqliteDbFilePath))
            {
                throw new ArgumentException($"Missing {SqliteDbFileEnvVarName} environment variable.");
            }

            // Connect to the database.
            //
            sqliteDbFilePath = Path.GetFullPath(sqliteDbFilePath);
            SqliteOpenMode dbMode = File.Exists(sqliteDbFilePath) ? SqliteOpenMode.ReadWrite : SqliteOpenMode.ReadWriteCreate;
            var connectionString = new SqliteConnectionStringBuilder
            {
                DataSource = new Uri(sqliteDbFilePath).AbsoluteUri,
                Mode = dbMode,
            };
            using var dbConnection = new SqliteConnection(connectionString.ToString());
            dbConnection.Open();

            // Make sure the schema is setup.
            //
            if (dbMode == SqliteOpenMode.ReadWriteCreate)
            {
                using var transaction = dbConnection.BeginTransaction(IsolationLevel.Serializable);
                SqliteCommand sqlCmd = dbConnection.CreateCommand();
                sqlCmd.CommandText = @"
                        CREATE TABLE IF NOT EXISTS CompileCommands
                        (
                            working_dir VARCHAR(255) NOT NULL,
                            file        VARCHAR(255) NOT NULL,
                            command     TEXT NOT NULL,
                            output      TEXT NULL,
                            PRIMARY KEY (working_dir, file)
                        )
                    ";
                sqlCmd.ExecuteNonQuery();
                transaction.Commit();
            }

            //
            // Process the arguments.
            //

            List<string> cmdArgs = new List<string>();
            foreach (string arg in args)
            {
                if (arg.StartsWith('@'))
                {
                    // Handle the rsp file args.
                    //
                    using StreamReader responseFile = File.OpenText(arg.TrimStart('@'));
                    string responseText = responseFile.ReadToEnd();
                    char[] whitespace = { ' ', '\t', '\n', '\r', '\v' };
                    cmdArgs.AddRange(responseText.Split(whitespace, StringSplitOptions.RemoveEmptyEntries));
                }
                else
                {
                    cmdArgs.Add(arg);
                }
            }

            // Parse output file from the args.
            //
            string outputFile = null;
            string[] outputFileTypes = new[] { ".obj" };
            for (int i = 0; i < cmdArgs.Count; i++)
            {
                if (cmdArgs[i] == "-o" && (i + 1) < cmdArgs.Count)
                {
                    outputFile = cmdArgs[i + 1];
                    if (outputFileTypes.Contains(Path.GetExtension(outputFile)))
                    {
                        throw new ArgumentException($"Unexpected output file type: '{outputFile}'");
                    }
                }
            }

            // Parse the target file from the args.
            // (assume it's the last one for now)
            //
            string sourceFile = cmdArgs.Last();
            string[] sourceFileExtensions = new[] { ".cpp", ".cxx", ".c" };
            if (!File.Exists(sourceFile))
            {
                throw new FileNotFoundException($"Source file '{sourceFile}' doesn't exist.");
            }
            else if (!sourceFileExtensions.Contains(Path.GetExtension(sourceFile)))
            {
                throw new ArgumentException($"Unexpected input file type: '{outputFile}'");
            }

            // Insert the command details to the database.
            //
            using (var transaction = dbConnection.BeginTransaction())
            {
                SqliteCommand sqlCmd = dbConnection.CreateCommand();

                if (outputFile != null)
                {
                    sqlCmd.CommandText = @"
                        REPLACE INTO CompileCommands (working_dir, file, command, output)
                        VALUES($working_dir, $file, $command, $output)
                    ";
                }
                else
                {
                    sqlCmd.CommandText = @"
                        REPLACE INTO CompileCommands (working_dir, file, command)
                        VALUES($working_dir, $file, $command)
                    ";
                }

                sqlCmd.Parameters.AddWithValue("$working_dir", System.Environment.CurrentDirectory);
                sqlCmd.Parameters.AddWithValue("$file", sourceFile);
                sqlCmd.Parameters.AddWithValue("$command", string.Join(' ', cmdArgs));

                if (outputFile != null)
                {
                    sqlCmd.Parameters.AddWithValue("$output", outputFile);
                }

                sqlCmd.ExecuteNonQuery();
                transaction.Commit();
            }

            dbConnection.Close();

            // Also output the command line used to invoke this process.
            //
            System.Console.Error.WriteLine(System.Environment.CommandLine);

            // Always report success.
            //
            return 0;
        }
    }
}