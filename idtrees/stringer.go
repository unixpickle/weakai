package idtrees

import (
	"bytes"
	"fmt"
	"strings"
)

// String returns a human-readable representation of
// the tree, using indentation to signify depth.
func (t *Tree) String() string {
	if t.Classification != nil {
		return classificationString(t.Classification)
	}

	var buf bytes.Buffer

	buf.WriteString(fmt.Sprintf("%v", t.Attr))
	buf.WriteRune('\n')

	split := t.ValSplit

	if t.NumSplit != nil {
		lessKey := fmt.Sprintf("<= %v", t.NumSplit.Threshold)
		greaterKey := fmt.Sprintf("> %v", t.NumSplit.Threshold)
		split = ValSplit{
			lessKey:    t.NumSplit.LessEqual,
			greaterKey: t.NumSplit.Greater,
		}
	}

	isFirst := true
	for value, subtree := range split {
		if !isFirst {
			buf.WriteRune('\n')
		}
		isFirst = false
		var subBuf bytes.Buffer
		subBuf.WriteRune(' ')
		subBuf.WriteString(fmt.Sprintf("%v", value))
		subBuf.WriteString(" -> ")
		subBuf.WriteString(subtree.String())
		buf.WriteString(strings.Replace(subBuf.String(), "\n", "\n  ", -1))
	}

	return buf.String()
}

func classificationString(m map[Class]float64) string {
	if len(m) == 0 {
		return "Unreachable"
	}
	var parts []string
	for key, frac := range m {
		parts = append(parts, fmt.Sprintf("%v=%.02f%%", key, frac*100))
	}
	return strings.Join(parts, " ")
}
