package pangu

import (
        "log"
        "bytes"
        "os/exec"
        "strings"
)

const pangu_root_path = "pangu://AT-20ANT/elasticdl/"
const pangu_client = "/pangu/pangu_cli.py"

// CopyRemoteFileToLocal copy the remote pangu file to lcoal filesystem.
// The Param remoteFilePath as a relative subpath of constant pangu_root_path.
func CopyRemoteFileToLocal(remoteFilePath string, localFilePath string) {
    fullRemoteFilePath := pangu_root_path + remoteFilePath
    cmd := exec.Command("python", pangu_client, "cp", fullRemoteFilePath, localFilePath)
    executeCmd(cmd)
}

// CopyLocalFileToRemote copy the local file to remote pangu filesystem.
// The Param remoteFilePath as a relative subpath of constant pangu_root_path.
func CopyLocalFileToRemote(localFilePath string, remoteFilePath string) {
    fullRemoteFilePath := pangu_root_path + remoteFilePath
    cmd := exec.Command("python", pangu_client, "cp", localFilePath, fullRemoteFilePath)
    executeCmd(cmd)
}

// RemoveRemoteFile remove the file in remote pangu filesystem.
// The Param remoteFilePath as a relative subpath of constant pangu_root_path.
func RemoveRemoteFile(remoteFilePath string) {
    fullRemoteFilePath := pangu_root_path + remoteFilePath
    cmd := exec.Command("python", pangu_client, "rm", fullRemoteFilePath)
    executeCmdWithY(cmd)
}

// CreateRemoteDir create new directory in remote pangu filesystem.
// The Param remoteDir as a relative subpath of constant pangu_root_path.
func CreateRemoteDir(remoteDir string) {
    fullRemoteDir := pangu_root_path + remoteDir
    cmd := exec.Command("python", pangu_client, "mkdir", fullRemoteDir)
    executeCmd(cmd)
}

func executeCmd(cmd *exec.Cmd) {
    var stdout, stderr bytes.Buffer
    cmd.Stdout = &stdout
    cmd.Stderr = &stderr
    cmd.Run()
    outStr, errStr := string(stdout.Bytes()), string(stderr.Bytes())
    if len(outStr) > 0 {
        log.Println(outStr)
    }
    if len(errStr) > 0 {
        log.Println(errStr)
    }
    if cmd.ProcessState.Success() == false {
        panic("Failed to execute cmd: " + strings.Join(cmd.Args, " "))
    }
}

func executeCmdWithY(cmd *exec.Cmd) {
    stdin, err := cmd.StdinPipe()
    if err != nil {
        panic("Failed to get stdin pipe, " + err.Error())
    }
    _, err = stdin.Write([]byte("Y"))
    if err != nil {
        panic("Failed to write value to stdin pipe, " + err.Error())
    }   
    stdin.Close()
    var stdout, stderr bytes.Buffer
    cmd.Stdout = &stdout
    cmd.Stderr = &stderr
    cmd.Run()
    outStr, errStr := string(stdout.Bytes()), string(stderr.Bytes())
    if len(outStr) > 0 {
        log.Println(outStr)
    }
    if len(errStr) > 0 {
        log.Println(errStr)
    }
    if cmd.ProcessState.Success() == false {
        panic("Failed to execute cmd: " + strings.Join(cmd.Args, " "))
    }
}
