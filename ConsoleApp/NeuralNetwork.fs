module NeuralNetwork

open MathNet.Numerics.Distributions
open MathNet.Numerics
open System
open MathNet.Numerics.LinearAlgebra
open LinearAlgebra.Matrix

// Byte value must be normalized to range from 0.01 to 0.99.
let normalizeFloat0 = 0.01
let normalizeFloat1 v : float = v * 0.98 + normalizeFloat0
let denormalizeFloat1 v : float = (v - normalizeFloat0) / 0.98
let randomMatrix outputRowsCount inputColsumnsCount = 
    // Scientifically proved optimal standard deviation.
    let stdDev = 1.0 / (sqrt (float inputColsumnsCount))
    DenseMatrix.init outputRowsCount inputColsumnsCount (fun _ _ -> Normal.Sample(0.0, stdDev) |> float)

// Matrixes of weightes of neuron connections beween neural network layers.
let randomMatrixList (layersSizeList : int list) = 
    layersSizeList |> Seq.pairwise |> Seq.map (fun (leftCount, rightCount) -> randomMatrix rightCount leftCount) |> List.ofSeq

/// <summary>Normalize neural network layer output by sigmoid function.</summary>
let logisticVector = Vector.map SpecialFunctions.Logistic

/// <summary>Converts neural network layer output to sigmiod-inverse input.</summary>
let logitVector = Vector.map SpecialFunctions.Logit

// Query trained neural network
let query (layers : Matrix<double> list) (inputs : Vector<float>) = List.fold (fun vector matrix -> matrix * vector |> logisticVector) inputs layers

let queryBack  (layers : Matrix<double> list) (outputs : Vector<float>) = 
    let scaleVectorToNormaziedFloat1 v = 
        let removeLow = v - (Vector.min v)
        let scaleTop = removeLow / (Vector.max removeLow)
        scaleTop |> Vector.map normalizeFloat1 
    let queryBackIter rightOutput matrix =
        let rightInput = rightOutput |> logitVector;
        let leftOutput = (transpose matrix) * rightInput |> scaleVectorToNormaziedFloat1
        leftOutput
    layers |> List.rev |> List.fold queryBackIter outputs

// Trains neural network by data sample
let trainSample (learningRate: float) (allLayerWeigthes: Matrix<float> list) (neuralNetworkInputs: Vector<float>) (neuralNetworkTartets: Vector<float>) =
    if List.isEmpty allLayerWeigthes then raise (ArgumentException("At least one layer required."))
    if allLayerWeigthes.Head.ColumnCount <> neuralNetworkInputs.Count then raise (ArgumentException("Input must be multiplied correctly."))
    if (List.last allLayerWeigthes).RowCount <> neuralNetworkTartets.Count then raise (ArgumentException("Target must be multiplied correctly."))

    let rec iter previousOutputs weightMatrixList =
        match weightMatrixList with
        | weightMatrix :: sublayers -> 
            let nextInputs = weightMatrix * previousOutputs
            let nextOutputs = nextInputs |> logisticVector
            let (nextErrors : Vector<float>), sublayersUpdated = iter nextOutputs sublayers
            
            let weightMatrixDelta = 
                DenseMatrix.ofColumns [nextErrors.PointwiseMultiply(nextOutputs).PointwiseMultiply(1.0 - nextOutputs)]
                * 
                DenseMatrix.ofRows [previousOutputs] 
                * 
                learningRate
            let weightMatrixUpdated = weightMatrix + weightMatrixDelta
            let previousErrors = (transpose weightMatrix) * nextErrors
            previousErrors, weightMatrixUpdated :: sublayersUpdated
        | [] -> 
            let outputErrors = neuralNetworkTartets - previousOutputs
            outputErrors, []
    iter neuralNetworkInputs allLayerWeigthes |> snd