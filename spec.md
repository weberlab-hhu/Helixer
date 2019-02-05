### features

#### feature start/end/at numbering

Features, in the final format (import still needs work...),
have just one coordinate that counts, (at/start).

That said, there is more to keep in mind about the positioning 
of the features. Specifically features essentially come in
three subtypes (start, status, end; e.g. transcription start site, 
status in raw transcript, transcription termination site). The 
positioning of these features is in keeping with the common
coordinate system: count from 0, start inclusive, end exclusive. 
So, the coding-start, is at the A, of the ATG, AKA the first
coding base pair; while in contrast, the coding-stop is
after the stop-codon, AKA, the first non-coding bp.

The statuses are used when for either a biological or a technical
reason, a transcript is split across two sequences or when 
part of the transcript is unknown (in an erroneous region).
When a single status is present (e.g. due to limited knowledge of 
true sequence/features), then it's positioning can be set to match 
as well as possible what is know. When, however, they are set as 
an UpDownPair at an artificial split, then they to should be 
considered start/end features, and essentially indicate a single bp at which
the status was noted. That is start (inclusively) on the last base pair on
sequence0, and end (exclusively) on the first bp of sequence1. 

##### reverse complement

Importantly, the coding-start should always point to the first
A, of ATG, regardless of strand. This means the numeric coordinates
have to change and unfortunately while one could take the
sequence [1, 4) + strand, literally pop in 1 and 4 as python coordinates
and get the sequence; the same is not going to work on the minus strand.
Instead: 

```
plus strand start 1, end 4
 0  1  2  3  4  5
.N [A .T .G )N .N
 |  |  |  |  |  |
 N. T. A. C. N. N.
```

To get the reverse complement of this on the minus strand, we set the
inclusive start to 3, and exclusive end to 0. Note this is now off by 
one from the python coordinates


```
plus strand start 1, end 4
 0  1  2  3  4  5
.N [A .T .G )N .N
 |  |  |  |  |  |
 N( T. A. C] N. N.
```