// -----------------------------------------------------------------------
// <copyright file="MlosAgentServer.cs" company="Microsoft Corporation">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root
// for license information.
// </copyright>
// -----------------------------------------------------------------------

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using CommandLine;

/*
Reduce some complexity while debugging.
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.Hosting;
*/

using Mlos.Core;

using Mlos.Model.Services;
using Mlos.Model.Services.Spaces;

using MlosOptimizer = Mlos.Model.Services.Client.BayesianOptimizer;

namespace Mlos.Agent.Server
{
    /// <summary>
    /// The MlosAgentServer acts as a simple external agent and shim helper to
    /// wrap the various communication channels (shared memory to/from the smart
    /// component, grpc to the optimizer, grpc from the notebooks).
    /// </summary>
    public static class MlosAgentServer
    {
        /* Reduce some complexity while debugging.
        /// <summary>
        /// Starts a grpc server listening for requests from the notebook to drive
        /// the agent interactively.
        /// </summary>
        /// <param name="args">unused.</param>
        /// <returns>grpc server task.</returns>
        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.ConfigureKestrel(options =>
                    {
                        // Setup a HTTP/2 endpoint without TLS.
                        //
                        options.ListenAnyIP(5000, o => o.Protocols = HttpProtocols.Http2);
                    });
                    webBuilder.UseStartup<GrpcServer.Startup>();
                });
                */

        /// <summary>
        /// The main external agent server.
        /// </summary>
        /// <param name="args">command line arguments.</param>
        public static void Main(string[] args)
        {
            // This switch must be set before creating the GrpcChannel/HttpClient.
            //
            // Make these the very first thing we do.
            //
            AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2Support", true);
            AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2UnencryptedSupport", true);

            string executableFilePath = null;
            Uri optimizerAddressUri = null;

            var cliOptsParseResult = CommandLine.Parser.Default.ParseArguments<CliOptions>(args)
                .WithParsed(parsedOptions =>
                {
                    executableFilePath = parsedOptions.Executable;
                    optimizerAddressUri = parsedOptions.OptimizerUri;
                });
            if (cliOptsParseResult.Tag == ParserResultType.NotParsed)
            {
                // CommandLine already prints the help text for us in this case.
                //
                Console.Error.WriteLine("Failed to parse command line options.");
                Environment.Exit(1);
            }

            // Check for the executable before setting up any shared memory to
            // reduce cleanup issues.
            //
            if (executableFilePath != null && !File.Exists(executableFilePath))
            {
                throw new FileNotFoundException($"ERROR: --executable '{executableFilePath}' does not exist.");
            }

            Console.WriteLine("Mlos.Agent.Server");
            TargetProcessManager targetProcessManager = null;

            // Connect to gRpc optimizer only if user provided an address in the command line.
            //
            if (optimizerAddressUri != null)
            {
                Console.WriteLine("Connecting to the Mlos.Optimizer");

                // This switch must be set before creating the GrpcChannel/HttpClient.
                //
                AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2UnencryptedSupport", true);

                // This populates a variable for the various settings registry
                // callback handlers to use (by means of their individual
                // AssemblyInitializers) to know how they can connect with the
                // optimizer.
                //
                // See Also: AssemblyInitializer.cs within the SettingsRegistry
                // assembly project in question.
                //
                MlosContext.OptimizerFactory = new MlosOptimizer.BayesianOptimizerFactory(optimizerAddressUri);
            }

            // hack for testing:
            // try to register an optimization problem, without even loading the shared memory or another assembly
            //
            IOptimizerProxy optimizerProxy = null;
            if (MlosContext.OptimizerFactory != null)
            {
                Hypergrid searchSpace = new Hypergrid("x", new DiscreteDimension("x", 0, 10));
                Hypergrid objectiveSpace = new Hypergrid("y", new ContinuousDimension("y", 0, 100));
                OptimizationProblem optimizationProblem = new OptimizationProblem
                {
                    ParameterSpace = searchSpace,
                    ContextSpace = null,
                    ObjectiveSpace = objectiveSpace,
                };
                optimizationProblem.Objectives.Add(new OptimizationObjective
                    {
                        Name = "y",
                        Minimize = false,
                    });

                Console.Error.WriteLine("Creating OptimizerProxy in Mlos.Agent.Server");
                optimizerProxy = MlosContext.OptimizerFactory.CreateRemoteOptimizer(optimizationProblem: optimizationProblem);
                Console.Error.WriteLine("Created OptimizerProxy in Mlos.Agent.Server");
            }

            // hacking: quit before mucking around with any of the shared memory, executable, their assemblies, etc.
            Environment.Exit(1);

            // Create (or open) the circular buffer shared memory before running the target process.
            //
            using var mainAgent = new MainAgent();
            mainAgent.InitializeSharedChannel();

            // Active learning mode.
            //
            // TODO: In active learning mode the MlosAgentServer can control the
            // workload against the target component.
            //
            if (executableFilePath != null)
            {
                Console.WriteLine($"Starting {executableFilePath}");
                targetProcessManager = new TargetProcessManager(executableFilePath: executableFilePath);
                targetProcessManager.StartTargetProcess();
            }
            else
            {
                Console.WriteLine("No executable given to launch.  Will wait for agent to connect independently.");
            }

            var cancellationTokenSource = new CancellationTokenSource();

            Task grpcServerTask = null; //// FIXME: CreateHostBuilder(Array.Empty<string>()).Build().RunAsync(cancellationTokenSource.Token);

            // Start the MainAgent message processing loop as a background thread.
            //
            // In MainAgent.RunAgent we loop on the shared memory control and
            // telemetry channels looking for messages and dispatching them to

            // their registered callback handlers.
            //
            // The set of recognized messages is dynamically registered using

            // the RegisterSettingsAssembly method which is called through the
            // handler for the RegisterAssemblyRequestMessage.
            //
            // Once registered, the SettingsAssemblyManager uses reflection to

            // search for an AssemblyInitializer inside those assemblies and
            // executes it in order to setup the message handler callbacks
            // within the agent.
            //
            // See Also: AssemblyInitializer.cs within the SettingsRegistry
            // assembly project in question.
            //
            Console.WriteLine("Starting Mlos.Agent");
            Task mlosAgentTask = Task.Factory.StartNew(
                () => mainAgent.RunAgent(),
                TaskCreationOptions.LongRunning);

            Task waitForTargetProcessTask = Task.Factory.StartNew(
                () =>
                {
                    if (targetProcessManager != null)
                    {
                        targetProcessManager.WaitForTargetProcessToExit();
                        targetProcessManager.Dispose();
                        mainAgent.UninitializeSharedChannel();
                    }
                },
                TaskCreationOptions.LongRunning);

            Console.WriteLine("Waiting for Mlos.Agent to exit");

            while (true)
            {
                Task.WaitAny(new[] { mlosAgentTask, waitForTargetProcessTask });

                if (mlosAgentTask.IsFaulted && targetProcessManager != null && !waitForTargetProcessTask.IsCompleted)
                {
                    // MlosAgentTask has failed, however the target process is still active.
                    // Terminate the target process and continue shutdown.
                    //
                    targetProcessManager.TerminateTargetProcess();
                    continue;
                }

                if (mlosAgentTask.IsCompleted && waitForTargetProcessTask.IsCompleted)
                {
                    // MlosAgentTask is no longer processing messages, and target process does no longer exist.
                    // Shutdown the agent.
                    //
                    break;
                }
            }

            // Print any exceptions if occured.
            //
            if (mlosAgentTask.Exception != null)
            {
                Console.WriteLine($"Exception: {mlosAgentTask.Exception}");
            }

            if (waitForTargetProcessTask.Exception != null)
            {
                Console.WriteLine($"Exception: {waitForTargetProcessTask.Exception}");
            }

            // Perform some cleanup.
            //
            waitForTargetProcessTask.Dispose();

            mlosAgentTask.Dispose();

            targetProcessManager?.Dispose();

            cancellationTokenSource.Cancel();
            grpcServerTask?.Wait();

            grpcServerTask?.Dispose();
            cancellationTokenSource.Dispose();

            Console.WriteLine("Mlos.Agent exited.");
        }

        /// <summary>
        /// The command line options for this application.
        /// </summary>
        private class CliOptions
        {
            [Option("executable", Required = false, Default = null, HelpText = "A path to an executable to start (e.g. 'target/bin/Release/SmartCache').")]
            public string Executable { get; set; }

            [Option("optimizer-uri", Required = false, Default = null, HelpText = "A URI to connect to the MLOS Optimizer service over GRPC (e.g. 'http://localhost:50051').")]
            public Uri OptimizerUri { get; set; }
        }
    }
}
