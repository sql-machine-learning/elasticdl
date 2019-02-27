package pangu

import (
        "log"
        "bytes"
        "os/exec"
        "strings"
)

const pangu_root_path = "pangu://${clusterName}/elasticdl/"
const pangu_client = "/pangu/pangu_cli.py"

// CopyRemoteFileToLocal copy the remote pangu file to lcoal filesystem.
// The Param remoteFilePath as a relative subpath of constant pangu_root_path.
func CopyRemoteFileToLocal(clusterName string, remoteFilePath string, localFilePath string) {
    fullRemoteFilePath := strings.Replace(pangu_root_path, "${clusterName}", clusterName, 1) + remoteFilePath
    cmd := exec.Command("python", pangu_client, "cp", fullRemoteFilePath, localFilePath)
    executeCmd(cmd)
}

// CopyLocalFileToRemote copy the local file to remote pangu filesystem.
// The Param remoteFilePath as a relative subpath of constant pangu_root_path.
func CopyLocalFileToRemote(clusterName string, localFilePath string, remoteFilePath string) {
    fullRemoteFilePath := strings.Replace(pangu_root_path, "${clusterName}", clusterName, 1) + remoteFilePath
    cmd := exec.Command("python", pangu_client, "cp", localFilePath, fullRemoteFilePath)
    executeCmd(cmd)
}

// RemoveRemoteFile remove the file in remote pangu filesystem.
// The Param remoteFilePath as a relative subpath of constant pangu_root_path.
func RemoveRemoteFile(clusterName string, remoteFilePath string) {
    fullRemoteFilePath := strings.Replace(pangu_root_path, "${clusterName}", clusterName, 1) + remoteFilePath
    cmd := exec.Command("python", pangu_client, "rm", fullRemoteFilePath)
    executeCmdWithY(cmd)
}

// CreateRemoteDir create new directory in remote pangu filesystem.
// The Param remoteDir as a relative subpath of constant pangu_root_path.
func CreateRemoteDir(clusterName string, remoteDir string) {
    fullRemoteDir := strings.Replace(pangu_root_path, "${clusterName}", clusterName, 1) + remoteDir
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
