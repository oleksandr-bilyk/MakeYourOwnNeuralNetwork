/// Analizes THE MNIST DATABASE of handwritten digits using neural network.
module MnistDatabaseNeuralNetwork
open NeuralNetwork
open MnistDatabase
open System
open MathNet.Numerics.LinearAlgebra.VectorExtensions
open MathNet.Numerics.LinearAlgebra

let minDigit = 0uy
let maxDigit = 9uy
let countDigit = maxDigit + 1uy |> int
let learningRateDefault = 0.3

let mnistLabeledDataNormalizedSequence (dataFiles : MnistDataFileNamesPair) =
    let imagesHeader, labeledImages, disposableComposite = mnistLabeledImageSequence dataFiles
    let labeledDataNormanized = labeledImages |> Seq.map (fun (label, data) -> label, (data |> Seq.map normanizeByte |> Seq.toList))
    imagesHeader, labeledDataNormanized, disposableComposite

let outputVectorTarget (digit : byte) =
    if digit < minDigit || digit > maxDigit then raise (ArgumentException(sprintf "Digit must be in range %i-%i." minDigit maxDigit))
    [for __ = 1uy to digit do yield normalizeFloat0] @ (normalizeFloat1 1.0)::[for __ = digit + 1uy to maxDigit do yield normalizeFloat0]

let predictedNumberProbabilitySortedDesc (output : float array) =
    if (output |> Seq.length) <> countDigit  then raise (ArgumentException(sprintf "Digit must be in range %i-%i." minDigit maxDigit))
    output |> Array.indexed |> Array.sortByDescending snd

let train (dataFilesForTraining : MnistDataFileNamesPair) (logTrainingProgress : int -> int -> unit) =
    let imagesHeader, dataSequence, disposableComposite = mnistLabeledDataNormalizedSequence dataFilesForTraining
    use __ = disposableComposite
    let inputCount = imagesHeader.RowsCount * imagesHeader.ColumnsCount
    let randomModel = randomMatrixListNew [inputCount;100;countDigit]

    let logIteration = logTrainingProgress imagesHeader.ImagesCount
    let learningIteration (iterationIndex, model) (target, data) = 
        let modelUpdated = train learningRateDefault model (vector data) (target |> outputVectorTarget |> vector)
        logIteration iterationIndex
        iterationIndex + 1, modelUpdated
    let __, trainedModel = dataSequence |> Seq.fold learningIteration (0, randomModel)
    trainedModel

let test (model: Matrix<float> list) (dataFilesForTesting : MnistDataFileNamesPair) =
    let imagesHeader, dataSequence, disposableComposite = mnistLabeledDataNormalizedSequence dataFilesForTesting
    use __ = disposableComposite

    let iter (target : byte, data : float list) = 
        let resultList = query model (vector data)
        let resultTop = resultList.ToArray() |> predictedNumberProbabilitySortedDesc
        ((target |> int) = (resultTop.[0] |> fst))
    let scoredCount = dataSequence |> Seq.filter iter |> Seq.length
    (scoredCount |> float) / (imagesHeader.ImagesCount |> float)
    