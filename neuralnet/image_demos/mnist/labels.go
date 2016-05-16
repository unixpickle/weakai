package main

import (
	"bufio"
	"io"
	"os"
)

type Labels []int

func ReadLabels(file string) (Labels, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	if _, err := r.Discard(8); err != nil {
		return nil, err
	}

	var res Labels
	for {
		if label, err := r.ReadByte(); err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		} else {
			res = append(res, int(label))
		}
	}

	return res, nil
}
