package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

func TestGRU(t *testing.T) {
	b := rnn.NewGRU(4, 2)
	NewChecker4In(b, b).FullCheck(t)
}
