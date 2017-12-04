#load "./packages/MathNet.Numerics.FSharp/MathNet.Numerics.fsx"
open MathNet.Numerics.Distributions
open MathNet.Numerics
open System
open MathNet.Numerics.LinearAlgebra

let randomMatrix rows cols = 
    let stdDev = 1.0 / (sqrt (float cols))
    DenseMatrix.init rows cols (fun _ _ -> Normal.Sample(0.0, stdDev) |> float)
let randomMatrixList layersCount nodesCount =  List.init layersCount (fun _ ->  randomMatrix nodesCount nodesCount)

let sigmoidVector = Vector.map (SpecialFunctions.Logistic >> float)
let query layers inputs = List.fold (fun vector matrix -> matrix * vector) inputs layers
let (<*>) v1 v2 = Vector.map2 (*) v1 v2

let train (learningRate: float) (allLayerWeigthes: Matrix<float> list) (neuralNetworkInputs: Vector<float>) (neuralNetworkTartets: Vector<float>) =
    if List.isEmpty allLayerWeigthes then raise (ArgumentException("At least one layer required."))
    let rec iter previousOutputs weightMatrixList =
        match weightMatrixList with
        | weightMatrix :: sublayers -> 
            let nextInputs = weightMatrix * previousOutputs
            let nextOutputs = nextInputs |> sigmoidVector
            let nextErrors, sublayersUpdated = iter nextOutputs sublayers
            
            let weightMatrixDelta = 
                DenseMatrix.ofColumns [nextErrors <*> nextOutputs <*> (1.0 - nextOutputs)]
                * 
                DenseMatrix.ofRows [previousOutputs] 
                * 
                learningRate
            let weightMatrixUpdated = weightMatrix + weightMatrixDelta
            let previousErrors = (Matrix.transpose weightMatrix) * nextErrors
            previousErrors, weightMatrixUpdated :: sublayersUpdated
        | [] -> 
            let outputErrors = neuralNetworkTartets - previousOutputs
            outputErrors, []
    iter neuralNetworkInputs allLayerWeigthes |> snd

// let train2 (learningRate: float) (wih: Matrix<float>) (who: Matrix<float>) (inputs: Vector<float>) (targets: Vector<float>) =
//     let hiddenInputs = wih * inputs
//     let hiddenOutputs = hiddenInputs |> sigmoidVector
//     let finalInputs = who * hiddenOutputs
//     let finalOutputs = finalInputs |> sigmoidVector
//     let outputError = targets - finalOutputs
//     let hiddenError = (Matrix.transpose who) * outputError
//     //let previousLayerValuesHO = 
//     let whoDelta = 
//         DenseMatrix.ofColumns [outputError <*> finalOutputs <*> (1.0 - finalOutputs)]
//         * 
//         DenseMatrix.ofRows [hiddenOutputs] 
//         * 
//         learningRate
//     let whoNew = who + whoDelta
//     let wihDelta = 
//         DenseMatrix.ofColumns [hiddenError <*> hiddenOutputs <*> (1.0 - hiddenOutputs)]
//         * 
//         DenseMatrix.ofRows [inputs] 
//         * 
//         learningRate
//     let wihNew = wih + wihDelta
//     whoNew, wihNew
    
let experementalModel = [
    DenseMatrix.ofRowList [[-0.59306787;  0.25274925; -0.32602831]; [-0.16685239;  0.22542431; -0.36808796]; [ 0.81883787;  1.29124618; -0.6584239 ]]
    DenseMatrix.ofRowList [[ 0.80042512;  0.35423876;  0.16241759]; [-1.52660991; -0.82271924;  0.23120044]; [ 0.66190966;  0.31868365;  0.39380777]]
]

let trainingTest = train 0.3 experementalModel (vector [1.0; 0.5; -1.5]) (vector [1.0; 0.5; -1.5])

type NeuralNetwork(inputNodes: int, hiddenNodes: int, outputNodes: int, learningRate: float) =
    
    let mutable weightsList = randomMatrixList 2 inputNodes
    
    member __.Train(inputs: Vector<float>, targets: Vector<float>) = 
        //train2 learningRate weightsInputHiden weightsHidenOutput inputs targets
        weightsList <- train learningRate weightsList inputs targets

    member __.Query(inputs : Vector<float>) = query weightsList inputs 
        

let n = NeuralNetwork(inputNodes = 3, hiddenNodes = 3, outputNodes = 3, learningRate = 0.3)
n.Query(vector [1.0; 0.5; -1.5])
