module NeuralNetwork

open MathNet.Numerics.Distributions
open MathNet.Numerics
open System
open MathNet.Numerics.LinearAlgebra

// Byte value must be normalized to range from 0.01 to 0.99.
let normalizeFloat0 = 0.01
let normalizeFloat1 v : float = v * 0.98 + normalizeFloat0
let normanizeByte (v : byte) = (float v) / (float Byte.MaxValue) |> normalizeFloat1
let randomMatrix outputRowsCount inputColsumnsCount = 
    // Scientifically proved optimal standard deviation.
    let stdDev = 1.0 / (sqrt (float inputColsumnsCount))
    DenseMatrix.init outputRowsCount inputColsumnsCount (fun _ _ -> Normal.Sample(0.0, stdDev) |> float)

// Matrixes of weightes of neuron connections beween neural network layers.
let randomMatrixListNew (layersSizeList : int list) = 
    layersSizeList |> Seq.pairwise |> Seq.map (fun (leftCount, rightCount) -> randomMatrix rightCount leftCount) |> List.ofSeq

// Normalize neural network layer output by sigmoid dunction.
let sigmoidVector = Vector.map (SpecialFunctions.Logistic >> float)

// Query trained neural network
let query (layers : Matrix<double> list) (inputs : Vector<float>) = List.fold (fun vector matrix -> matrix * vector |> sigmoidVector) inputs layers

// Trains neural network by data sample
let train (learningRate: float) (allLayerWeigthes: Matrix<float> list) (neuralNetworkInputs: Vector<float>) (neuralNetworkTartets: Vector<float>) =
    if List.isEmpty allLayerWeigthes then raise (ArgumentException("At least one layer required."))
    if allLayerWeigthes.Head.ColumnCount <> neuralNetworkInputs.Count then raise (ArgumentException("Input must be multiplied correctly."))
    if (List.last allLayerWeigthes).RowCount <> neuralNetworkTartets.Count then raise (ArgumentException("Target must be multiplied correctly."))

    let rec iter previousOutputs weightMatrixList =
        match weightMatrixList with
        | weightMatrix :: sublayers -> 
            let nextInputs = weightMatrix * previousOutputs
            let nextOutputs = nextInputs |> sigmoidVector
            let (nextErrors : Vector<float>), sublayersUpdated = iter nextOutputs sublayers
            
            let weightMatrixDelta = 
                DenseMatrix.ofColumns [nextErrors.PointwiseMultiply(nextOutputs).PointwiseMultiply(1.0 - nextOutputs)]
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