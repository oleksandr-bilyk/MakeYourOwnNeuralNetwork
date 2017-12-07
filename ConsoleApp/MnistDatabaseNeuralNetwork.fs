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

// Experementally taken value.
let learningRateDefault = 0.2

// Experementally taken value to get balance between underfeet and overfeeding
let learningEpochsDefault = 4

let normalizeLabeledImageSequence labeledImages = labeledImages |> Seq.map (fun (label, data) -> label, (data |> Seq.map normanizeByte |> Seq.toList))

let mnistLabeledDataNormalizedSequence (dataFiles : MnistDataFileNamesPair) =
    let readFromBegin, disposableComposite = mnistLabeledImageData dataFiles
    let imagesHeader, labeledImages = readFromBegin ()
    let labeledDataNormanized = labeledImages |> normalizeLabeledImageSequence
    imagesHeader, labeledDataNormanized, disposableComposite

let outputVectorTarget digit =
    if digit < minDigit || digit > maxDigit then raise (ArgumentException(sprintf "Digit must be in range %i-%i." minDigit maxDigit))
    [for __ = 1uy to digit do yield normalizeFloat0] @ (normalizeFloat1 1.0)::[for __ = digit + 1uy to maxDigit do yield normalizeFloat0]

let trainByDataSeq model logIteration data =
    let learningIteration (iterationIndex, model) (target, data) = 
        let modelUpdated = trainSample learningRateDefault model (vector data) (target |> outputVectorTarget |> vector)
        logIteration iterationIndex
        iterationIndex + 1, modelUpdated
    let __, trainedModel = data |> Seq.fold learningIteration (0, model)
    trainedModel

let train learningEpochs (dataFilesForTraining : MnistDataFileNamesPair) (totalSamplesProgress : int -> int -> unit) epochsProgress =
    let readDataFromBegin, disposableComposite = mnistLabeledImageData dataFilesForTraining 
    use __ = disposableComposite

    let getInitialData readDataFromBegin = 
        let header, __ = readDataFromBegin ()
        let getInitialRandomModel imageSize = randomMatrixListNew [imageSize.Height * imageSize.Width;100;countDigit]
        header, getInitialRandomModel header.Size
    let header, randomModel = getInitialData readDataFromBegin
    epochsProgress 0 randomModel
    let getDataSequence () = 
        let __, dataSequence = readDataFromBegin ()
        dataSequence |> normalizeLabeledImageSequence
    
    let logTrainingProgressTotalInitialize = totalSamplesProgress (header.ImagesCount * learningEpochs)
    let logIteration epochIndex iterationIdexInEpoch = logTrainingProgressTotalInitialize (epochIndex * header.ImagesCount + iterationIdexInEpoch)
    let rec trainingEpoch epochIndex model =
        if (epochIndex >= learningEpochs) then model
        else 
            let logIterationForEpoch = logIteration epochIndex
            let modelNew = getDataSequence () |> trainByDataSeq model logIterationForEpoch
            epochsProgress (epochIndex + 1) modelNew
            trainingEpoch (epochIndex + 1) modelNew
    trainingEpoch 0 randomModel

let trainDefaultEpochs dataFilesForTraining logTrainingProgress = train learningEpochsDefault dataFilesForTraining logTrainingProgress

let test (model: Matrix<float> list) (dataFilesForTesting : MnistDataFileNamesPair) =
    let readFromBegin, disposableComposite = mnistLabeledImageData dataFilesForTesting 
    use __ = disposableComposite
    let imagesHeader, dataSequence = readFromBegin ()

    let predictedNumberProbabilitySortedDesc (output : float array) =
        if (output |> Seq.length) <> countDigit  then raise (ArgumentException(sprintf "Digit must be in range %i-%i." minDigit maxDigit))
        output |> Array.indexed |> Array.sortByDescending snd

    let iter (target : byte, data : float list) = 
        let resultList = query model (vector data)
        let resultTop = resultList.ToArray() |> predictedNumberProbabilitySortedDesc
        ((target |> int) = (resultTop.[0] |> fst))
    let scoredCount = dataSequence |> normalizeLabeledImageSequence |> Seq.filter iter |> Seq.length
    (scoredCount |> float) / (imagesHeader.ImagesCount |> float)
    