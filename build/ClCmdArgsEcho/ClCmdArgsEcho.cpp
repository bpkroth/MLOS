//*********************************************************************
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root
// for license information.
//
// @File: ClCmdArgsEcho.cpp
//
// Purpose:
//      <description>
//
// Notes:
//      <special-instructions>
//
//********************************************************************

#include <iostream>

// ClCmdArgsEcho is a simple program to aid hooking into the ClangCompile task
// and targets that currently ship with MSBuild so that we can capture the
// commands that *would* be used to build the targets with clang.exe without
// actually needing to build them (in case that isn't actually supported by the
// target yet).
// 
// The intent is to dump those commands to a clang compilation database
// (compile_commands.json) for use with CodeBook's clang frontend plugin so
// that we can execute a parse-only pass on target's source files in order to
// ingest the code structures for analysis with CodeBook.
// 
int main(int argc, char* argv[])
{
	// TODO: Handle response files.

	// Just prints the arguments it was passed.
	// TODO: Make it output to a compile_commands.json file.
	//
	for (int i=1; i<argc; i++) // skip the first arg - it's just the name of this exe
	{
		std::cout << argv[i] << " ";
	}
	std::cout << std::endl;

	// Always return success.
	//
	return 0;
}
