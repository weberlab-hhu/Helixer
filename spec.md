### features

#### types
Feature types will now be broken down into two pieces, namely
type (transcribed, coding, intron, trans_intron, error) and
bearing (start, end, open_status, close_status, point).

##### new types:
* transcribed: primarily transcription start & termination sites
* coding: primarily start and stop codon
* intron / trans intron: primarily donor site, acceptor site
* error: mark start/end of region when the provided annotation is
in question.

##### bearing:
* start: start of a region, inclusive
  * transcription start site (1st transcribed bp)
  * start codon, the A of the ATG
  * donor splice site, first bp of intron
* end: end of a region, exclusive (i.e. start of one there after)
  * transcription termination site (1st non-transcribed bp)
  * stop codon, first non-coding bp, e.g. the N in TGAN
  * acceptor splice site (1st bp that is part of final transcript)
* open/close status:
  
  for defining a _region_ these will work the same as start/end.
  So every bp between intron: open_status and intron: close_status
  defines an intron. However, these can be used when there is incomplete
  information (e.g. the assembled sequence started in an intron, so the donor 
  splice site is missing). If however, the status is set in the middle of a
  sequence, this should generally be accompanied by an 'error mask' that 
  indicates the approximate region with an unclear identity / where the missing
  start/end feature may occur.
* point:
  
  anything occurring at a single point / not defining a region.
  OK, this has no current usage, but could be used, e.g. to mark
  an expected small mistake (SNP/deletion) in the assembly as erroneous.  


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
 0  1  2  3  4  5
.N [A .T .G )N .N
 |  |  |  |  |  |
 N. T. A. C. N. N.
```

To get the reverse complement of this on the minus strand, we set the
inclusive start to 3, and exclusive end to 0. Note this is now off by 
one from the python coordinates


```
 0  1  2  3  4  5
.N [A .T .G )N .N
 |  |  |  |  |  |
 N( T. A. C] N. N.
```

##### differences vs gff
Cheat sheet for how the Features compare to the gff (in particular any discrepancy
between the closest coordinate in the gff, and the now standardized, consistent coordinate).

First and last for gff are reported as they are typically in gff (coordinate sorted),
so reverse to the interpretation when on the - strand. 

Plus strand (+)

| Common Name  | GFF | GFF start | GFF end |type| bearing| position |
| -------------|:----| ---------:|--------:|:---|:-------|--------:|
| TSS, Transcription start site      | start 1st exon  |x| |transcribed|start|x - 1|
| TTS, Transcription termination site| end last exon   | |x|transcribed|end  |x    |
| 1st bp of start codon              | start 1st CDS   |x| |coding     |start|x - 1|
| last bp of stop codon              | end last CDS    | |x|coding     |end  |x    |
| donor splice site (5' of intron)   |end non-last exon| |x|intron     |start|x    |
| acceptor splice site (3' of intron)|start 2nd+ exon  |x| |intron     |end  |x - 1|


Minus strand (-)

| Common Name  | GFF | GFF start | GFF end |type| bearing| position|
| -------------|:----| ---------:|--------:|:---|:-------|--------:|
| TSS, Transcription start site      | end last exon   | |x|transcribed|start|x - 1|
| TTS, Transcription termination site| start 1st exon  |x| |transcribed|end  |x - 2|
| 1st bp of start codon              | end last CDS    | |x|coding     |start|x - 1|
| last bp of stop codon              | start 1st CDS   |x| |coding     |end  |x - 2|
| donor splice site (5' of intron)   |start 2nd+ exon  |x| |intron     |start|x - 2|
| acceptor splice site (3' of intron)|end non-last exon| |x|intron     |end  |x - 1|

