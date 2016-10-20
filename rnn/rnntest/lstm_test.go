package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

func TestLSTM(t *testing.T) {
	b := rnn.NewLSTM(4, 2)
	NewChecker4In(b, b).FullCheck(t)
}
