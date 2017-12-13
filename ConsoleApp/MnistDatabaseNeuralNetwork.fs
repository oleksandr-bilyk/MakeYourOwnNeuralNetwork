/// Analizes THE MNIST DATABASE of handwritten digits using neural network.
module MnistDatabaseNeuralNetwork

open NeuralNetwork
open MnistDatabase
open System
open MathNet.Numerics.LinearAlgebra.VectorExtensions
open MathNet.Numerics.LinearAlgebra
open MnistDatabaseExtraction
open System.IO

let minDigit = 0uy
let maxDigit = 9uy
let countDigit = maxDigit + 1uy |> int

// Experementally taken value.
let learningRateDefault = 0.1

let byteToNNInput (v : byte) = (float v) / (float Byte.MaxValue) |> normalizeFloat1
let nnImputToByte (v : float) = (v |> denormalizeFloat1) * (float Byte.MaxValue) |> byte

let normalizeLabeledImageSequence labeledImages = labeledImages |> Seq.map (fun (label, data) -> label, (data |> Seq.map byteToNNInput |> Seq.toList))

let outputVectorTarget digit =
    if digit < minDigit || digit > maxDigit then raise (ArgumentException(sprintf "Digit must be in range %i-%i." minDigit maxDigit))
    [for __ = 1uy to digit do yield normalizeFloat0] @ (normalizeFloat1 1.0)::[for __ = digit + 1uy to maxDigit do yield normalizeFloat0]
    |> vector

type TrainingDurationParameters = { learningEpochs : int; takeItems : int option }

type ChannelDescription = {
    PredictItemsCount : int -> int
    Collect : (byte * byte[]) seq -> (byte * byte[]) seq
}

let dataSecuenceCollectingMapping = List.map (fun c -> c.Collect) >> List.fold ( >> ) id

let trainWithChanels header dataSeq (channels : ChannelDescription list) totalProgressUpdate =
    let getInitialRandomModel imageSize = randomMatrixList [imageSize.Height * imageSize.Width;200;countDigit]
    let randomModel = getInitialRandomModel header.Size
    let totalImageCount = channels |> List.map (fun c -> c.PredictItemsCount) |> List.fold (fun c f -> f c) header.ImagesCount
    let totalDataSeq = dataSeq |> dataSecuenceCollectingMapping channels |> normalizeLabeledImageSequence

    let updateIteration = totalProgressUpdate totalImageCount
    let learningIteration (iterationIndex, model) (target, data) = 
        let modelUpdated = trainSample learningRateDefault model (vector data) (target |> outputVectorTarget)
        updateIteration iterationIndex
        iterationIndex + 1L, modelUpdated
    totalDataSeq |> Seq.fold learningIteration (0L, randomModel) |> snd

let epochsChanel epochsCount = { 
    PredictItemsCount = epochsCount |> (*)
    Collect = (fun sourceSeq -> seq { 1 .. epochsCount} |> Seq.collect (fun _ -> sourceSeq))
}

// Experementally taken value to get balance between underfeet and overfeeding
let learningEpochsDefault = epochsChanel 4

let takeFirstChannel number = {
    PredictItemsCount = min number;
    Collect = Seq.truncate number
}

let rotateDataImage angelDegree = (fun size -> rotateImageData angelDegree size.Width size.Height)
let defaultRotationAmplitude = 10.0F
let mutationProviders = [
    (fun __ -> id) // no rotation
    rotateDataImage defaultRotationAmplitude
    rotateDataImage -defaultRotationAmplitude
]

let imageMutationChannel imageSize = {
    PredictItemsCount = mutationProviders |> List.length  |> (*)
    Collect = (fun sourceSeq -> 
        mutationProviders |> Seq.collect (fun rotationProvider -> 
            let rotation = rotationProvider imageSize
            sourceSeq |> Seq.map (fun (label, data) -> label, (rotation data))
        )
    )
}

let trainDefaultEpochs dataFilesForTraining logTrainingProgress = 
    let header, readFromBegine, mnistDisposableComposit = mnistLabeledImageDataExt dataFilesForTraining 
    use __ = mnistDisposableComposit

    trainWithChanels header readFromBegine [ takeFirstChannel 100000; imageMutationChannel header.Size; learningEpochsDefault ] logTrainingProgress

let test (model: Matrix<float> list) (dataFilesForTesting : MnistDataFileNamesPair) =
    let imagesHeader, readFromBegine, mnistDisposableComposit = mnistLabeledImageDataExt dataFilesForTesting 
    use __ = mnistDisposableComposit

    let predictedNumberProbabilitySortedDesc (output : float array) =
        if (output |> Seq.length) <> countDigit  then raise (ArgumentException(sprintf "Digit must be in range %i-%i." minDigit maxDigit))
        output |> Array.indexed |> Array.sortByDescending snd

    let iter (target : byte, data : float list) = 
        let resultList = query model (vector data)
        let resultTop = resultList.ToArray() |> predictedNumberProbabilitySortedDesc
        ((target |> int) = (resultTop.[0] |> fst))
    let scoredCount = readFromBegine |> normalizeLabeledImageSequence |> Seq.filter iter |> Seq.length
    (scoredCount |> float) / (imagesHeader.ImagesCount |> float)

let generateAndSaveNumbersPareidolia dataFilesForTraining folder = 
    let header, readFromBegine, mnistDisposableComposit = mnistLabeledImageDataExt dataFilesForTraining 
    use __ = mnistDisposableComposit

    let model = trainWithChanels header readFromBegine [ (takeFirstChannel 1000) ] (fun _ _ -> ())
    let saveImagesSized = saveImageByteLabeled folder header.Size
    [minDigit .. maxDigit] 
    |> List.map 
        (fun label -> 
            let numberNNOutput = outputVectorTarget label
            let imageNNInputDream = queryBack model numberNNOutput
            let dreamImageByteArray = imageNNInputDream |> Seq.map nnImputToByte |> Array.ofSeq
            label, dreamImageByteArray
        )        
    |> List.iter saveImagesSized

let extractMnistDatabaseLabeledImages (dataFiles : MnistDataFileNamesPair) destinationFolder =
    let imagesHeader, readFromBegine, mnistDisposableComposit = mnistLabeledImageDataExt dataFiles
    use __ = mnistDisposableComposit

    let imageFileName (index : int) label = 
        let fileName = sprintf "%s-%i.png" ((index + 1).ToString("00000")) label
        Path.Combine(destinationFolder, fileName)
    let imageBySize = saveImage imagesHeader.Size.Width imagesHeader.Size.Height
    
    readFromBegine |> Seq.iteri (fun i (label, imageData) -> imageBySize imageData (imageFileName i label))