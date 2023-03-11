# JWNMF

This is an **Unofficial** implementation of method proposed by Tang in [1].

## Sample Data

In this repository, we've placed a test data which is a sythensized graph generated using method 
proposed in [2] and simulated cascades of posts exchanged between social network nodes. Underlying
graph in this data has 100 nodes and we've simulated 200 i.i.d. cascades on that. To use this to
for testing proposed method in [1], we've randomly deleted some links and cascade participations.

## How to run

Convert data

```Python
python converter/convert.py -d <dataset>
```

Run Code

```Python
python main.py -d <dataset>
```

## Parameters

## References

[1] Tang, M. (2022). A Joint Weighted Nonnegative Matrix Factorization Model via Fusing Attribute 
Information for Link Prediction. In: Chenggang, Y., Honggang, W., Yun, L. (eds) Mobile Multimedia 
Communications. MobiMedia 2022. Lecture Notes of the Institute for Computer Sciences, Social 
Informatics and Telecommunications Engineering, vol 451. Springer, Cham. 
<https://doi.org/10.1007/978-3-031-23902-1_15>

[2] Lancichinetti, A., Fortunato, S., & Radicchi, F. (2008). Benchmark graphs for testing 
community detection algorithms. Physical review E, 78(4), 046110.
