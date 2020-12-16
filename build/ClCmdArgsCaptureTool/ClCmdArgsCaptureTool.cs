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
    using System.Data;
    using System.IO;
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
                using (var transaction = dbConnection.BeginTransaction(IsolationLevel.Serializable))
                {
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
            }

            // Parse the arguments.

            string cppFile = "TODO: Parse the target source file from the args (and convert it into an absolute path use cwd).";

            // TODO: Handle the rsp file args.
            string cmdArgs = string.Join(" ", args);

            // TODO: Parse the output file from the args.
            string outputFile = null;

            // Insert the command details to the database.

            using (var transaction = dbConnection.BeginTransaction())
            {
                SqliteCommand sqlCmd = dbConnection.CreateCommand();

                if (outputFile == null)
                {
                    sqlCmd.CommandText = @"
                        REPLACE INTO CompileCommands (working_dir, file, command) VALUES($working_dir, $file, $command)
                    ";
                }
                else
                {
                    sqlCmd.CommandText = @"
                        REPLACE INTO CompileCommands (working_dir, file, command, output) VALUES($working_dir, $file, $command, $output)
                    ";
                }

                sqlCmd.Parameters.AddWithValue("$working_dir", System.Environment.CurrentDirectory);
                sqlCmd.Parameters.AddWithValue("$file", cppFile);
                sqlCmd.Parameters.AddWithValue("$command", cmdArgs);

                if (outputFile == null)
                {
                    sqlCmd.Parameters.AddWithValue("$output", outputFile);
                }

                sqlCmd.ExecuteNonQuery();
                transaction.Commit();
            }

            dbConnection.Close();
            dbConnection.Dispose();

            // Also output the command line used to invoke this process.
            //
            System.Console.WriteLine(System.Environment.CommandLine);

            // Always report success.
            //
            return 0;
        }
    }
}