#load "./packages/MathNet.Numerics.FSharp/MathNet.Numerics.fsx"
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics

let randomMatrix rows cols = 
    let stdDev = 1.0 / (sqrt (double cols))
    DenseMatrix.init rows cols (fun _ _ -> Normal.Sample(0.0, stdDev) |> double)

let sigmoidVector = Vector.map (SpecialFunctions.Logistic >> double)
let layerQuery matrix input = matrix * input |> sigmoidVector
let mulltiLayerQuery matrixList input = 
    List.fold (fun vector matrix -> layerQuery matrix vector) input matrixList

type NeuralNetwork(inputNodes: int, hiddenNodes: int, outputNodes: int, learningRate: float) =
    let weightsInputHiden = randomMatrix inputNodes hiddenNodes
    let weightsHidenOutput = randomMatrix hiddenNodes outputNodes
    let weightsList = [weightsInputHiden; weightsHidenOutput]
    
    member __.Train() = ()
    member __.Query(inputList : double list) = 
        vector inputList |> mulltiLayerQuery weightsList

let n = NeuralNetwork(inputNodes = 3, hiddenNodes = 3, outputNodes = 3, learningRate = 0.3)
SpecialFunctions.Logistic -10.0