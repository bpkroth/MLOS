# ClCmdArgsCaptureTool

`ClCmdArgsCaptureTool` is a simple program to aid hooking into the
ClangCompile task and targets that currently ship with MSBuild so that we can
capture the commands that *would* be used to build the targets with `clang`
without actually needing to build them (in case that isn't actually supported
by the target yet).

The intent is to dump those commands to a clang compilation database
(`compile_commands.json`) for use with CodeBook's `clang` frontend plugin so
that we can execute a parse-only pass on target's source files in order to
ingest the code structures for analysis with CodeBook.

## Usage

### Steps to integrate with an existing build system

TODO

### Populating the `compile_commands.db` SQLite database file

TODO

### Exporting to a `compile_commands.json`

TODO
