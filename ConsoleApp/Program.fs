// Learn more about F# at http://fsharp.org

open System


open System

open System.Threading

open System.Reflection

open System.Diagnostics

let userDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
let p = new Process()
p.StartInfo.FileName <- "jupyter"
p.StartInfo.Arguments <- "notebook"
p.StartInfo.WorkingDirectory <- userDir

[<EntryPoint>]
let main argv =
    argv |> ignore
    printfn "Hello World from F#!"
    Console.ReadKey() |> ignore
    0 // return an integer exit code
