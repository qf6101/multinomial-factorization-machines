# Multinomial Factorization Machines

## Brief Description

This project implements the binomial and multinomial factorization machines (FM and MFM). Factorization machines are a generic approach that combines the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain. Please refer to [\[Steffen Rendle (2010)\]](Factorization Machines) for more detail.

This implementation is based on Spark 2.0.0 as compared with the famous standalone implementation known as [libfm](http://www.libfm.org/).

## Binomial FM Usage

Binomial FM is the same as the standard version. Please refer to [src/test/scala/io/github/qf6101/mfm/factorization/binomial/FmSuite.scala](src/test/scala/io/github/qf6101/mfm/factorization/binomial/FmSuite.scala
) for detailed usage.

## Multinomial FM Usage

Multinomial FM is desinged for multiclass classification problem which uses softmax as hypothesis. Please refer to [src/test/scala/io/github/qf6101/mfm/factorization/multinomial/MfmSuite.scala
](src/test/scala/io/github/qf6101/mfm/factorization/multinomial/MfmSuite.scala
) for detailed usage.

