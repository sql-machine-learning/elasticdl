package pangu

import (
        "testing"
        "os"
        "io/ioutil"
)

const clusterName = "AT-20ANT"

func TestUploadAndDownloadFile(t *testing.T) {
    const remoteFilePath = "test/tmp_file.txt"
    const localFilePath = "tmp_file.txt"

    s := []byte("hi, elasticdl")
    ioutil.WriteFile(localFilePath, s, os.ModeAppend)

    CopyLocalFileToRemote(clusterName, localFilePath, remoteFilePath)
    CopyRemoteFileToLocal(clusterName, remoteFilePath, localFilePath)
    RemoveRemoteFile(clusterName, remoteFilePath)

    err := os.Remove(localFilePath)
    if err != nil {
        t.Error(err.Error())
    }
}

func TestCreateRemoteDir(t *testing.T) {
    CreateRemoteDir(clusterName, "tmp_dir")
    RemoveRemoteFile(clusterName, "tmp_dir")
}
