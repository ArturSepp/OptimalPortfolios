# Supplemental third-party notices

OptimalPortfolios source is MIT-licensed. Dependencies and their bundled native
libraries retain their own licences. This file supplies the complete OpenBLAS
and LAPACK notices identified during the 19 October 2026 Windows CPython 3.12
x64 base-install review. It does not replace dependency notices, repair an
upstream wheel, or certify other versions, platforms, extras or redistribution
of an entire environment.

The reviewed PyPI wheels contain these OpenBLAS 0.3.33 libraries while their own
wheel licence directories contain only their top-level project licence:

- `scs==3.3.1`, `scs.libs/openblas-ea5706aaa7e88ea76f350faf4181c5e3.dll`
  (SHA-256 `ea5706aaa7e88ea76f350faf4181c5e3393817a1ce1cd800501bf3ee21f205e8`).
  The tagged SCS Windows workflow obtains OpenBLAS 0.3.33 from conda-forge.
- `sparsediffpy==0.6.1`,
  `sparsediffpy.libs/openblas-0a8e50eb27b63dddae4f0314d8e43548.dll`
  (SHA-256 `0a8e50eb27b63dddae4f0314d8e43548aa242a87006f19c5b8d0bbbe768def54`).
  The tagged Windows workflow uses vcpkg; this binary identifies itself as
  `OpenBLAS 0.3.33 NO_LAPACK NO_LAPACKE NO_AFFINITY generic`.

OpenBLAS and its LAPACK source notices are reproduced without changing holders
or permission terms from the upstream `v0.3.33` files below. The LAPACK notice
applies where LAPACK is included; the reviewed SparseDiffPy binary is built
without LAPACK. Retain these notices with any corresponding distributed binary.
The individual SCS MIT and SparseDiffPy Apache-2.0 notices still apply separately.

Sources:
- https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.33/LICENSE
- https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.33/lapack-netlib/LICENSE
- https://github.com/bodono/scs-python/blob/3.3.1/.github/workflows/build.yml
- https://github.com/SparseDifferentiation/SparseDiffPy/blob/v0.6.1/.github/workflows/build-and-publish.yml

## OpenBLAS 0.3.33

```text
Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of 
      its contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

## LAPACK included by OpenBLAS 0.3.33

```text
Copyright (c) 1992-2023 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved.
Copyright (c) 2000-2023 The University of California Berkeley. All
                        rights reserved.
Copyright (c) 2006-2023 The University of Colorado Denver.  All rights
                        reserved.

$COPYRIGHT$

Additional copyrights may follow

$HEADER$

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

The copyright holders provide no reassurances that the source code
provided does not infringe any patent, copyright, or any other
intellectual property rights of third parties.  The copyright holders
disclaim any liability to any recipient for claims brought against
recipient by any third party for infringement of that parties
intellectual property rights.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```
