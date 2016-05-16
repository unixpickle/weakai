package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"

	"github.com/unixpickle/weakai/neuralnet"
)

func ReadSamples(file string) ([]*neuralnet.Tensor3, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	if _, err := r.Discard(4); err != nil {
		return nil, err
	}

	var params [3]uint32

	for i := 0; i < 3; i++ {
		if err := binary.Read(r, binary.BigEndian, &params[i]); err != nil {
			return nil, err
		}
	}

	count := int(params[0])
	width := int(params[1])
	height := int(params[2])

	tensors := make([]*neuralnet.Tensor3, count)
	for j := 0; j < count; j++ {
		var buffer bytes.Buffer
		limited := io.LimitedReader{R: r, N: int64(width * height)}
		if n, err := io.Copy(&buffer, &limited); err != nil {
			return nil, err
		} else if n < int64(width*height) {
			return nil, errors.New("not enough data for image")
		}
		tensor := neuralnet.NewTensor3(width, height, 1)
		data := buffer.Bytes()
		for i := range tensor.Data {
			tensor.Data[i] = 1 - (float64(data[i]) / 255.0)
		}
		tensors[j] = tensor
	}

	return tensors, nil
}
