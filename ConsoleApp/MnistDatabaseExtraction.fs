/// Extracts THE MNIST DATABASE of handwritten digits (http://yann.lecun.com/exdb/mnist/) content as user human recognized images.
module MnistDatabaseExtraction

open System.IO
open SixLabors.ImageSharp
open MnistDatabase

let saveImage width height (data : byte[]) fileName = 
    use img = new Image<Rgba32>(width, height)
    for i = 0 to (height - 1) do
        for j = 0 to (width - 1) do
            let byte = data.[i * width + j]
            let pixel = Rgba32(byte, byte, byte)
            img.[j, i] <- pixel
    use file = File.Create(fileName)
    img.SaveAsPng(file)
    ()

let extractMnistDatabaseLabeledImages (dataFiles : MnistDataFileNamesPair) destinationFolder =
    let imagesHeader, labeledImages, dataStreamsDisposable = mnistLabeledImageSequence dataFiles
    use __ = dataStreamsDisposable

    let imageFileName (index : int) label = 
        let fileName = sprintf "%s-%i.png" ((index + 1).ToString("00000")) label
        Path.Combine(destinationFolder, fileName)
    let imageBySize = saveImage imagesHeader.ColumnsCount imagesHeader.RowsCount
    
    labeledImages |> Seq.iteri (fun i (label, imageData) -> imageBySize imageData (imageFileName i label))