#!/usr/bin/nu

$env.config.table.mode = "compact"

def clearall [] { clear; run-external "printf" "'\\e[3J'" }

# Compile/execute a binary with cargo and plot it with python.
def main [
    --no-compute (-C) # Don't run the executable
    --no-plot (-P) # Don't run the plotting script
    --alt-plot (-p): path # Specify a plotting script
    --clear (-c) # Clear the terminal before doing anything
    --list (-l) # List available executables/plotting scripts
    exec?: string # Name of executable/plotting script
]: nothing -> nothing {
    let execs = (open Cargo.toml | get bin | get name)
    let plots = (ls -s src | get name | where ($it | str ends-with "_plot.py"))

    if $clear { clearall }

    if $list or ($exec == null) {
        $execs
        | filter {|ex| $"($ex)_plot.py" in $plots }
        | each {|ex| print $ex }
        return
    }

    if not $no_compute {
        if $exec in $execs {
            cargo run --release --bin $exec
        } else {
            error make { msg: $"missing executable ($exec)" }
        }
    }
    if not $no_plot {
        if $alt_plot != null {
            python $alt_plot
        } else if $"($exec)_plot.py" in $plots {
            python $"src/($exec)_plot.py"
        } else {
            error make { msg: $"missing plot script ($exec)_plot.py" }
        }
    }

    return
}

