module MnistDatabaseExtraction

open System.IO
open SixLabors.ImageSharp

open MnistDatabase
open System

let image width height (data : byte[]) fileName = 
    use img = new Image<Rgba32>(width, height)
    for i = 0 to (height - 1) do
        for j = 0 to (width - 1) do
            let byte = data.[i * width + j]
            let pixel = Rgba32(byte, byte, byte)
            img.[j, i] <- pixel
    use file = File.Create(fileName)
    img.SaveAsPng(file)
    ()

let mnistLabeledImages imageDatabaseFileName labelDatabaseFileName destinationFolder =
    use imagesDataStream = openUncompressedStream imageDatabaseFileName
    use labelsDataStream = openUncompressedStream labelDatabaseFileName

    let imagesHeader, imagesDataSeq = mnistImagesData imagesDataStream
    let labelsCount, labelSeq = mnistLabelsData labelsDataStream

    //Seq.iter (fun s -> System.Diagnostics.Debug.WriteLine(sprintf "%i" s)) labelSeq

    if imagesHeader.ImagesCount <> labelsCount then raise (Exception("Images and labals count must be equal."))

    let imageFileName (index : int) label = 
        let fileName = sprintf "%s-%i.png" ((index + 1).ToString("00000")) label
        Path.Combine(destinationFolder, fileName)
    let combinedSequence = Seq.zip labelSeq imagesDataSeq
    let imageBySize = image imagesHeader.ColumnsCount imagesHeader.RowsCount
    
    Seq.iteri (fun i (label, imageData) -> imageBySize imageData (imageFileName i label)) combinedSequence

// let a = readMnistDatabaseFromCompressedFile fileName |> Array.ofSeq
// printfn "%A" a.Length