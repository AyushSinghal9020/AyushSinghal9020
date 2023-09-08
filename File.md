# Attention is All you Need

If we search `transformers-architechture`/`attention is all you need`. We get a very complicated image 

<img src = 'https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png' width = 300>

But it is actually simple then most of the architechtures came before. To understand this deeply we fisrt need to understand the concept of `Multi Head Attention`. So what does this Multi Head Attention actually mean

<img src = 'https://machinelearningmastery.com/wp-content/uploads/2022/03/dotproduct_1.png' width = 400>

Lets assume we have a word like `Optimus Prime`. How can we represent this word in numbers. 

One way is to assign distinct number to each character like 
```
[1 , 2 , 3 , 4 , 5 , 6 , 0 , 2 , 7 , 4 , 5 , 8]
```
Same like we can encode words in large corpus where the maximum range of elements can be $60$ having `small case letters`/`capital case letters`/`punctuations` and much more

But this thing losses `positional inforamtion`. To counter this porblem we can do another thing 

|Optimus|Prime|
|---|---|
|1|0
|0|1

This method gives us a vector like 
```
[
    [1 , 0] , 
    [0 , 1]
]
```

where the dimension can be `[length of the sentence , number of total words]`.

This type of encoding is called `Sentence Peice Tokenization`

So lets assume we have an array like this 
```
[0,0,0,0,0,0,0,1,0,0,0]
```
We make $3$ copies of this array and send through different `Linear Layers`. For now take `Linear Layer` as a big matrix, that helps in the `Linear Transformation` of `arrays`

The $3$ Linear Layers are named as `Query`/`Key`/`Value`

We first only compute the first $2$ `Linear Layers` - `Key`/`Query`. Then we multiply their outputs. If they are multi-dimensional, they are multiplied as corss products of `Matrix`. These multipled values can show a lot of variance and thus these are scaled. Further these are passed through a `Softmax Layer`, which chnages the values to set of probablities. 

These are then multiplied with the output of `Value - Linear Layer`. 

We call this as `Multi Head Attentions`. One thing to note here is sometimes we break the arrays into peices and then do the whole process. We also combine those peices in the end. This not only helps us in less computation but also helps to identify/focus words locally.

We further pass the whole input through another `Linear Layer`. Then we pass them through `Layer Normalization` followed by another `Linear Layer` followed by another `Layer Normalization`

One thing to add here is we add the values before the `Linear Layer` to the `Layer Normalization`. which changes the process name to `Add & Layer Norm`

This completes our `Encoder Block` of the diagram

The `Decoder Block` becomes simple now
* Pass the `Q`/`K`/`V`
* Then the `Multi Head Self Attention`
* Then the `Add & Layer Norm`
* Then another `Multi Head Self Attention`
* Then another `Add & Layer Norm`
* Then the `Linear Layer`
* Then another `Add & Layer Norm`

One thing we skipped here is `Masking`. The first `Multi Head Self Attention` consists of `Masked Inputs`. cause at the time of decoding the `Sentence`. we do not know the words that will come afterwords.

Another thing to note here is the second `MHSA` consists of values from the `Encoder Block as well`

This whole architechture completes our `Transformer Architechture`/`Attention is All you Need`
