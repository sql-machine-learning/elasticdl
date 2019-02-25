package pangu

import (
        "testing"
        "os"
        "io/ioutil"
)

func TestUploadAndDownloadFile(t *testing.T) {
    const remoteFilePath = "test/tmp_file.txt"
    const localFilePath = "tmp_file.txt"

    s := []byte("hi, elasticdl")
    ioutil.WriteFile(localFilePath, s, os.ModeAppend)

    CopyLocalFileToRemote(localFilePath, remoteFilePath)
    CopyRemoteFileToLocal(remoteFilePath, localFilePath)
    RemoveRemoteFile(remoteFilePath)

    err := os.Remove(localFilePath)
    if err != nil {
        t.Error(err.Error())
    }
}

func TestCreateRemoteDir(t *testing.T) {
    CreateRemoteDir("tmp_dir")
    RemoveRemoteFile("tmp_dir")
}
