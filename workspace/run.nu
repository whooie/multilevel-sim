#!/usr/bin/nu

$env.config.table.mode = "compact"

def clearall [] { clear; run-external "printf" "'\\e[3J'" }

def main [] { }

# Compile/execute a binary with cargo and run all associated python scripts.
def "main exec" [
  --compute-only (-C) # Only run the executable
  --scripts-only (-S) # Only run the associated scripts
  --clear (-c) # Clear the terminal before doing anything
  --list (-l) # List available target executables
  target?: string # Name of executable
]: nothing -> nothing {
  let execs = (open Cargo.toml | get bin | get name)
  let scriptdirs = (ls -s src | where type == "dir" | get name)
  let targets = (
    $execs | where $it in $scriptdirs
    | append ($scriptdirs | where $it in $execs)
    | uniq
    | sort
  )
  if not (($target == null) or ($target in $targets)) {
    error make { msg: $"invalid target ($target)" }
  }

  if $clear { clearall }
  if $list or ($target == null) {
    $targets | each {|t| print $t }
    return
  }
  if not $scripts_only {
    cargo run --release --bin $target
  }
  if not $compute_only {
    ls $"src/($target)/*.py" | get name | each {|script| python $script }
  }

  return
}

