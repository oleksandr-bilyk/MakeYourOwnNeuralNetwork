// THE MNIST DATABASE reading module
module MnistDatabase

open System
open System.IO
open System.IO.Compression

let openUncompressedStream fileName : Stream =
    let fileStream = File.OpenRead(fileName)
    new GZipStream(fileStream, CompressionMode.Decompress) :> Stream

let readNetworkInt (binaryReader : BinaryReader) = 
    let intRaw = binaryReader.ReadInt32()
    // According to THE MNIST DATABASE documentation (http://yann.lecun.com/exdb/mnist/)
    // All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. 
    System.Net.IPAddress.NetworkToHostOrder(intRaw)

type ImagesFileHeader = { ImagesCount : int; RowsCount : int; ColumnsCount : int }

let mnistImagesData stream : (ImagesFileHeader * byte[] seq) = 
    let binaryReader = new BinaryReader(stream)
    let readNextInt () = readNetworkInt binaryReader
    let magicNumber = readNextInt ()
    if magicNumber <> 2051 then raise (Exception("Magic number expected."))
    let imagesNumber = readNextInt ()
    let rowsNumber = readNextInt ()
    let columnsNumber = readNextInt ()
    let imagesSeq = seq { for _i = 1 to imagesNumber do yield binaryReader.ReadBytes((rowsNumber * columnsNumber)) }
    ({ ImagesCount = imagesNumber; RowsCount = rowsNumber; ColumnsCount = columnsNumber }, imagesSeq)

let mnistLabelsData stream : (int * byte seq) = 
    let binaryReader = new BinaryReader(stream)
    let readNextInt () = readNetworkInt binaryReader
    let magicNumber = readNextInt ()
    if magicNumber <> 2049 then raise (Exception("Magic number expected."))
    let itemsCount = readNextInt ()
    let labelsSeq = seq { for _i = 1 to itemsCount do yield binaryReader.ReadByte() }
    itemsCount, labelsSeq

// THE MNIST DATABASE of handwritten digits (http://yann.lecun.com/exdb/mnist/) are stored in pairs of custom binary files. 
// Images file contains array of image data.
// Labels file contains number digits depicted on images.
type MnistDataFileNamesPair = { Labels: string; Images: string }

let mnistLabeledImageSequence (dataFiles : MnistDataFileNamesPair) =
    let imagesDataStream = openUncompressedStream dataFiles.Images
    try
        let labelsDataStream = openUncompressedStream dataFiles.Labels
        try
            let imagesHeader, imagesDataSeq = mnistImagesData imagesDataStream
            let labelsCount, labelSeq = mnistLabelsData labelsDataStream

            if imagesHeader.ImagesCount <> labelsCount then raise (Exception("Images and labals count must be equal."))
          
            let combinedSequence = Seq.zip labelSeq imagesDataSeq

            let disposableComposite = { 
                new IDisposable with member __.Dispose() = [imagesDataStream :> IDisposable; labelsDataStream :> IDisposable] |> List.iter (fun d -> d.Dispose())
            }
            imagesHeader, combinedSequence, disposableComposite
        with | _ ->
            labelsDataStream.Dispose()
            reraise()
    with | _ ->
        imagesDataStream.Dispose()
        reraise()