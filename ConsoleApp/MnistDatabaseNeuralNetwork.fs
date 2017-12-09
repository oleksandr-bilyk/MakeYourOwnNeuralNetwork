/// Analizes THE MNIST DATABASE of handwritten digits using neural network.
module MnistDatabaseNeuralNetwork

open NeuralNetwork
open MnistDatabase
open System
open MathNet.Numerics.LinearAlgebra.VectorExtensions
open MathNet.Numerics.LinearAlgebra
open MnistDatabaseExtraction

let minDigit = 0uy
let maxDigit = 9uy
let countDigit = maxDigit + 1uy |> int

// Experementally taken value.
let learningRateDefault = 0.1

// Experementally taken value to get balance between underfeet and overfeeding
let learningEpochsDefault = 4

let byteToNNInput (v : byte) = (float v) / (float Byte.MaxValue) |> normalizeFloat1
let nnImputToByte (v : float) = (v |> denormalizeFloat1) * (float Byte.MaxValue) |> byte

let normalizeLabeledImageSequence labeledImages = labeledImages |> Seq.map (fun (label, data) -> label, (data |> Seq.map byteToNNInput |> Seq.toList))

let mnistLabeledDataNormalizedSequence (dataFiles : MnistDataFileNamesPair) =
    let readFromBegin, disposableComposite = mnistLabeledImageData dataFiles
    let imagesHeader, labeledImages = readFromBegin ()
    let labeledDataNormanized = labeledImages |> normalizeLabeledImageSequence
    imagesHeader, labeledDataNormanized, disposableComposite

let outputVectorTarget digit =
    if digit < minDigit || digit > maxDigit then raise (ArgumentException(sprintf "Digit must be in range %i-%i." minDigit maxDigit))
    [for __ = 1uy to digit do yield normalizeFloat0] @ (normalizeFloat1 1.0)::[for __ = digit + 1uy to maxDigit do yield normalizeFloat0]
    |> vector
let trainByDataSeq model logIteration data =
    let learningIteration (iterationIndex, model) (target, data) = 
        let modelUpdated = trainSample learningRateDefault model (vector data) (target |> outputVectorTarget)
        logIteration iterationIndex
        iterationIndex + 1, modelUpdated
    let __, trainedModel = data |> Seq.fold learningIteration (0, model)
    trainedModel

type TrainingDurationParameters = { learningEpochs : int; takeItems : int option }
let train trainingDurationParameters (dataFilesForTraining : MnistDataFileNamesPair) (totalSamplesProgress : int -> int -> unit) epochsProgress =
    let readDataFromBegin, disposableComposite = mnistLabeledImageData dataFilesForTraining 
    use __ = disposableComposite

    let getInitialData readDataFromBegin = 
        let header, __ = readDataFromBegin ()
        let getInitialRandomModel imageSize = randomMatrixList [imageSize.Height * imageSize.Width;200;countDigit]
        header, getInitialRandomModel header.Size
    let header, randomModel = getInitialData readDataFromBegin
    epochsProgress 0 randomModel
    let getDataSequence () = 
        let __, dataSequence = readDataFromBegin ()
        dataSequence |> normalizeLabeledImageSequence
    
    let imagesCountForProgress, itemsTake = 
        match trainingDurationParameters.takeItems with 
        | None -> header.ImagesCount, (id)
        | Some c -> 
            let checkedCount = min c header.ImagesCount
            checkedCount, Seq.take checkedCount
    let logTrainingProgressTotalInitialize = totalSamplesProgress (imagesCountForProgress * trainingDurationParameters.learningEpochs)
    let logIteration epochIndex iterationIdexInEpoch = logTrainingProgressTotalInitialize (epochIndex * imagesCountForProgress + iterationIdexInEpoch)
    let rec trainingEpoch epochIndex model =
        if (epochIndex >= trainingDurationParameters.learningEpochs) then model
        else 
            let logIterationForEpoch = logIteration epochIndex
            let modelNew = getDataSequence () |> itemsTake |> trainByDataSeq model logIterationForEpoch
            epochsProgress (epochIndex + 1) modelNew
            trainingEpoch (epochIndex + 1) modelNew
    let resultModel = trainingEpoch 0 randomModel
    header.Size, resultModel

let trainDefaultEpochs dataFilesForTraining logTrainingProgress = 
    train { learningEpochs = learningEpochsDefault; takeItems = None } dataFilesForTraining logTrainingProgress

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
    
let generateAndSaveNumbersPareidolia dataFilesForTraining folder = 
    let imageSize, model = train { learningEpochs = 1; takeItems = Some 1000 } dataFilesForTraining (fun _ _ -> ()) (fun _ _ -> ())
    let saveImagesSized = saveImageByteLabeled folder imageSize
    [minDigit .. maxDigit] 
    |> List.map 
        (fun label -> 
            let numberNNOutput = outputVectorTarget label
            let imageNNInputDream = queryBack model numberNNOutput
            let dreamImageByteArray = imageNNInputDream |> Seq.map nnImputToByte |> Array.ofSeq
            label, dreamImageByteArray
        )        
    |> List.iter saveImagesSized