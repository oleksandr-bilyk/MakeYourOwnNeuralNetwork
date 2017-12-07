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

type ImageSize = { Height : int; Width : int }
type ImagesFileHeader = { ImagesCount : int; Size : ImageSize }

let mnistImagesHeader stream =
    let binaryReader = new BinaryReader(stream)
    let readNextInt () = readNetworkInt binaryReader
    let magicNumber = readNextInt ()
    if magicNumber <> 2051 then raise (Exception("Magic number expected."))
    let imagesNumber = readNextInt ()
    let rowsNumber = readNextInt ()
    let columnsNumber = readNextInt ()
    { ImagesCount = imagesNumber; Size = { Height = rowsNumber; Width = columnsNumber }}

let mnistImagesSeq header stream = 
    let binaryReader = new BinaryReader(stream)
    let imageByteSize = header.Size.Height * header.Size.Width
    seq { 
        for i = 1 to header.ImagesCount do 
            yield binaryReader.ReadBytes(imageByteSize)
            if i = header.ImagesCount then stream.Close()
    }

let mnistImagesData stream = 
    let header = mnistImagesHeader stream
    let sequence = mnistImagesSeq header stream
    header, sequence

let mnistLabelsHeader stream =
    let binaryReader = new BinaryReader(stream)
    let readNextInt () = readNetworkInt binaryReader
    let magicNumber = readNextInt ()
    if magicNumber <> 2049 then raise (Exception("Magic number expected."))
    readNextInt ()

let mnistLabelsSeq itemsCount stream = 
    let binaryReader = new BinaryReader(stream)
    seq { 
        for i = 1 to itemsCount do 
            yield binaryReader.ReadByte() 
            if i = itemsCount then stream.Close()
    }

let mnistLabelsData stream : (int * byte seq) = 
    let itemsCount = mnistLabelsHeader stream
    let sequence = mnistLabelsSeq itemsCount stream
    itemsCount, sequence

// THE MNIST DATABASE of handwritten digits (http://yann.lecun.com/exdb/mnist/) are stored in pairs of custom binary files. 
// Images file contains array of image data.
// Labels file contains number digits depicted on images.
type MnistDataFileNamesPair = { Labels: string; Images: string }

// Data files may be opened once but read all sequence multiple times.
let mnistLabeledImageData (dataFiles : MnistDataFileNamesPair) =
    let imagesComplressedStream = File.OpenRead(dataFiles.Images) 
    try
        let labelsCompressedStream = File.OpenRead(dataFiles.Labels)
        try
            let readFromBegine () = 
                imagesComplressedStream.Position <- 0L
                let imagesDataStream = new GZipStream(imagesComplressedStream, CompressionMode.Decompress, true)
                let imagesHeader, imagesDataSeq = mnistImagesData imagesDataStream
                
                labelsCompressedStream.Position <- 0L
                let labelsDataStream = new GZipStream(labelsCompressedStream, CompressionMode.Decompress, true)
                let labelsCount, labelSeq = mnistLabelsData labelsDataStream
                
                if imagesHeader.ImagesCount <> labelsCount then raise (Exception("Images and labals count must be equal."))
          
                let combinedSequence = Seq.zip labelSeq imagesDataSeq

                imagesHeader, combinedSequence

            let disposableComposite = { 
                new IDisposable with member __.Dispose() = [imagesComplressedStream :> IDisposable; labelsCompressedStream :> IDisposable] |> List.iter (fun d -> d.Dispose())
            }

            readFromBegine, disposableComposite
        with | _ ->
            labelsCompressedStream.Dispose()
            reraise()
    with | _ ->
        imagesComplressedStream.Dispose()
        reraise()    