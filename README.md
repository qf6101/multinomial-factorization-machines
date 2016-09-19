# Multinomial Factorization Machines

## Brief Description

This project implements the binomial and multinomial factorization machines. Factorization machines are a generic approach that combines the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain. Please refer to [\[Steffen Rendle (2010)\]](http://www.inf.uni-konstanz.de/~rendle/pdf/Rendle2010FM.pdf) for more detail.

This implementation is based on Spark 2.0.0 as compared with the famous standalone implementation known as [libfm](http://www.libfm.org/). Some auxiliary codes (e.g., the optimization and Logging) were adopted from Spark's private internals.

## Binomial Factorization Machines (FM)

FM is designed for binary-class classification problem as the standard [libfm](http://www.libfm.org/). Please refer to [src/test/scala/io/github/qf6101/mfm/factorization/binomial/FmSuite.scala](src/test/scala/io/github/qf6101/mfm/factorization/binomial/FmSuite.scala
) for detailed usage.

> Note: The implementation takes the labels as +1/-1.

## Mutinomial Factorization Machines (MFM)

MFM is desinged for multi-class classification problem which uses softmax as hypothesis. Please refer to [src/test/scala/io/github/qf6101/mfm/factorization/multinomial/MfmSuite.scala
](src/test/scala/io/github/qf6101/mfm/factorization/multinomial/MfmSuite.scala
) for detailed usage.

> Note: The implementation takes the labels as 0, 1, 2, etc.