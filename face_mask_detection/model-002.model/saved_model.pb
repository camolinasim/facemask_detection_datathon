вн
╤г
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878ь╦
Е
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*!
shared_nameconv2d_26/kernel
~
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*'
_output_shapes
:╚*
dtype0
u
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nameconv2d_26/bias
n
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes	
:╚*
dtype0
|
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
И╜2* 
shared_namedense_23/kernel
u
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel* 
_output_shapes
:
И╜2*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:2*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:2*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
У
Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*(
shared_nameAdam/conv2d_26/kernel/m
М
+Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/m*'
_output_shapes
:╚*
dtype0
Г
Adam/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*&
shared_nameAdam/conv2d_26/bias/m
|
)Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/m*
_output_shapes	
:╚*
dtype0
К
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
И╜2*'
shared_nameAdam/dense_23/kernel/m
Г
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m* 
_output_shapes
:
И╜2*
dtype0
А
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:2*
dtype0
И
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_24/kernel/m
Б
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:2*
dtype0
А
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:*
dtype0
У
Adam/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*(
shared_nameAdam/conv2d_26/kernel/v
М
+Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/v*'
_output_shapes
:╚*
dtype0
Г
Adam/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*&
shared_nameAdam/conv2d_26/bias/v
|
)Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/v*
_output_shapes	
:╚*
dtype0
К
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
И╜2*'
shared_nameAdam/dense_23/kernel/v
Г
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v* 
_output_shapes
:
И╜2*
dtype0
А
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:2*
dtype0
И
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_24/kernel/v
Б
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:2*
dtype0
А
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
┬.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¤-
valueє-BЁ- Bщ-
Ъ
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
м
0iter

1beta_1

2beta_2
	3decay
4learning_ratemhmi$mj%mk*ml+mmvnvo$vp%vq*vr+vs
*
0
1
$2
%3
*4
+5
*
0
1
$2
%3
*4
+5
 
н
5layer_regularization_losses
		variables

trainable_variables
6layer_metrics

7layers
8non_trainable_variables
regularization_losses
9metrics
 
\Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
:layer_regularization_losses
	variables
trainable_variables
;layer_metrics

<layers
=non_trainable_variables
regularization_losses
>metrics
 
 
 
н
?layer_regularization_losses
	variables
trainable_variables
@layer_metrics

Alayers
Bnon_trainable_variables
regularization_losses
Cmetrics
 
 
 
н
Dlayer_regularization_losses
	variables
trainable_variables
Elayer_metrics

Flayers
Gnon_trainable_variables
regularization_losses
Hmetrics
 
 
 
н
Ilayer_regularization_losses
	variables
trainable_variables
Jlayer_metrics

Klayers
Lnon_trainable_variables
regularization_losses
Mmetrics
 
 
 
н
Nlayer_regularization_losses
 	variables
!trainable_variables
Olayer_metrics

Players
Qnon_trainable_variables
"regularization_losses
Rmetrics
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
н
Slayer_regularization_losses
&	variables
'trainable_variables
Tlayer_metrics

Ulayers
Vnon_trainable_variables
(regularization_losses
Wmetrics
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
н
Xlayer_regularization_losses
,	variables
-trainable_variables
Ylayer_metrics

Zlayers
[non_trainable_variables
.regularization_losses
\metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
1
0
1
2
3
4
5
6
 

]0
^1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	_total
	`count
a	variables
b	keras_api
D
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

a	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

f	variables
}
VARIABLE_VALUEAdam/conv2d_26/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Т
serving_default_conv2d_26_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
з
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_26_inputconv2d_26/kernelconv2d_26/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_45883
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
░

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_46193
П
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_26/kernelconv2d_26/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_26/kernel/mAdam/conv2d_26/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_46284√╘
░
л
C__inference_dense_24_layer_call_and_return_conditional_losses_45735

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2:::O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
ї
╛
-__inference_sequential_12_layer_call_fn_45982

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_458412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Х>
П
__inference__traced_save_46193
file_prefix/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2bf808b4408544689e04a748e446fbc0/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameМ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names└
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesН
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*▀
_input_shapes═
╩: :╚:╚:
И╜2:2:2:: : : : : : : : : :╚:╚:
И╜2:2:2::╚:╚:
И╜2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:╚:!

_output_shapes	
:╚:&"
 
_output_shapes
:
И╜2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:╚:!

_output_shapes	
:╚:&"
 
_output_shapes
:
И╜2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::-)
'
_output_shapes
:╚:!

_output_shapes	
:╚:&"
 
_output_shapes
:
И╜2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
Ф'
·
H__inference_sequential_12_layer_call_and_return_conditional_losses_45919

inputs,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identityИ┤
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:╚*
dtype02!
conv2d_26/Conv2D/ReadVariableOp├
conv2d_26/Conv2DConv2Dinputs'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚*
paddingVALID*
strides
2
conv2d_26/Conv2Dл
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp▒
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚2
conv2d_26/BiasAddЗ
activation_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         ╚2
activation_26/Relu╧
max_pooling2d_26/MaxPoolMaxPool activation_26/Relu:activations:0*0
_output_shapes
:         ╚*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPoolu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"    И^  2
flatten_12/Constе
flatten_12/ReshapeReshape!max_pooling2d_26/MaxPool:output:0flatten_12/Const:output:0*
T0*)
_output_shapes
:         И╜2
flatten_12/Reshapey
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_11/dropout/Constл
dropout_11/dropout/MulMulflatten_12/Reshape:output:0!dropout_11/dropout/Const:output:0*
T0*)
_output_shapes
:         И╜2
dropout_11/dropout/Mul
dropout_11/dropout/ShapeShapeflatten_12/Reshape:output:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape╫
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*)
_output_shapes
:         И╜*
dtype021
/dropout_11/dropout/random_uniform/RandomUniformЛ
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_11/dropout/GreaterEqual/yь
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:         И╜2!
dropout_11/dropout/GreaterEqualв
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:         И╜2
dropout_11/dropout/Castи
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*)
_output_shapes
:         И╜2
dropout_11/dropout/Mul_1к
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
И╜2*
dtype02 
dense_23/MatMul/ReadVariableOpд
dense_23/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         22
dense_23/Reluи
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
dense_24/MatMul/ReadVariableOpг
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/MatMulз
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpе
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/BiasAdd|
dense_24/SoftmaxSoftmaxdense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_24/Softmaxn
IdentityIdentitydense_24/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         :::::::W S
/
_output_shapes
:         
 
_user_specified_nameinputs
и
c
*__inference_dropout_11_layer_call_fn_46044

inputs
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_456792
StatefulPartitionedCallР
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:         И╜2

Identity"
identityIdentity:output:0*(
_input_shapes
:         И╜22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
▐
}
(__inference_dense_23_layer_call_fn_46069

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_457082
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*0
_input_shapes
:         И╜::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
л
м
D__inference_conv2d_26_layer_call_and_return_conditional_losses_45623

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:╚*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╨
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_45684

inputs

identity_1\
IdentityIdentityinputs*
T0*)
_output_shapes
:         И╜2

Identityk

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:         И╜2

Identity_1"!

identity_1Identity_1:output:0*(
_input_shapes
:         И╜:Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
н
L
0__inference_max_pooling2d_26_layer_call_fn_45609

inputs
identityь
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_456032
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ь
F
*__inference_dropout_11_layer_call_fn_46049

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_456842
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         И╜2

Identity"
identityIdentity:output:0*(
_input_shapes
:         И╜:Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
═
·
H__inference_sequential_12_layer_call_and_return_conditional_losses_45948

inputs,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identityИ┤
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:╚*
dtype02!
conv2d_26/Conv2D/ReadVariableOp├
conv2d_26/Conv2DConv2Dinputs'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚*
paddingVALID*
strides
2
conv2d_26/Conv2Dл
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp▒
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚2
conv2d_26/BiasAddЗ
activation_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         ╚2
activation_26/Relu╧
max_pooling2d_26/MaxPoolMaxPool activation_26/Relu:activations:0*0
_output_shapes
:         ╚*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPoolu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"    И^  2
flatten_12/Constе
flatten_12/ReshapeReshape!max_pooling2d_26/MaxPool:output:0flatten_12/Const:output:0*
T0*)
_output_shapes
:         И╜2
flatten_12/ReshapeЗ
dropout_11/IdentityIdentityflatten_12/Reshape:output:0*
T0*)
_output_shapes
:         И╜2
dropout_11/Identityк
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
И╜2*
dtype02 
dense_23/MatMul/ReadVariableOpд
dense_23/MatMulMatMuldropout_11/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_23/MatMulз
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_23/BiasAdd/ReadVariableOpе
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         22
dense_23/Reluи
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
dense_24/MatMul/ReadVariableOpг
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/MatMulз
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpе
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/BiasAdd|
dense_24/SoftmaxSoftmaxdense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_24/Softmaxn
IdentityIdentitydense_24/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         :::::::W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┴
a
E__inference_flatten_12_layer_call_and_return_conditional_losses_45659

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    И^  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         И╜2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         И╜2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:X T
0
_output_shapes
:         ╚
 
_user_specified_nameinputs
Р
╟
-__inference_sequential_12_layer_call_fn_45856
conv2d_26_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall╡
StatefulPartitionedCallStatefulPartitionedCallconv2d_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_458412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_26_input
Р
╟
-__inference_sequential_12_layer_call_fn_45816
conv2d_26_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall╡
StatefulPartitionedCallStatefulPartitionedCallconv2d_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_458012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_26_input
■
~
)__inference_conv2d_26_layer_call_fn_46001

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_456232
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
√
╦
H__inference_sequential_12_layer_call_and_return_conditional_losses_45841

inputs
conv2d_26_45821
conv2d_26_45823
dense_23_45830
dense_23_45832
dense_24_45835
dense_24_45837
identityИв!conv2d_26/StatefulPartitionedCallв dense_23/StatefulPartitionedCallв dense_24/StatefulPartitionedCallЯ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_26_45821conv2d_26_45823*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_456232#
!conv2d_26/StatefulPartitionedCallП
activation_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_456442
activation_26/PartitionedCallФ
 max_pooling2d_26/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_456032"
 max_pooling2d_26/PartitionedCall■
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_12_layer_call_and_return_conditional_losses_456592
flatten_12/PartitionedCall°
dropout_11/PartitionedCallPartitionedCall#flatten_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_456842
dropout_11/PartitionedCallо
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_23_45830dense_23_45832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_457082"
 dense_23/StatefulPartitionedCall┤
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_45835dense_24_45837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_457352"
 dense_24/StatefulPartitionedCallч
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┴
a
E__inference_flatten_12_layer_call_and_return_conditional_losses_46017

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    И^  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         И╜2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         И╜2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:X T
0
_output_shapes
:         ╚
 
_user_specified_nameinputs
л
м
D__inference_conv2d_26_layer_call_and_return_conditional_losses_45992

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:╚*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┌
d
H__inference_activation_26_layer_call_and_return_conditional_losses_45644

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╚2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:X T
0
_output_shapes
:         ╚
 
_user_specified_nameinputs
▐
╜
#__inference_signature_wrapper_45883
conv2d_26_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallconv2d_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_455972
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_26_input
╩
∙
H__inference_sequential_12_layer_call_and_return_conditional_losses_45752
conv2d_26_input
conv2d_26_45634
conv2d_26_45636
dense_23_45719
dense_23_45721
dense_24_45746
dense_24_45748
identityИв!conv2d_26/StatefulPartitionedCallв dense_23/StatefulPartitionedCallв dense_24/StatefulPartitionedCallв"dropout_11/StatefulPartitionedCallи
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallconv2d_26_inputconv2d_26_45634conv2d_26_45636*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_456232#
!conv2d_26/StatefulPartitionedCallП
activation_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_456442
activation_26/PartitionedCallФ
 max_pooling2d_26/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_456032"
 max_pooling2d_26/PartitionedCall■
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_12_layer_call_and_return_conditional_losses_456592
flatten_12/PartitionedCallР
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_456792$
"dropout_11/StatefulPartitionedCall╢
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_23_45719dense_23_45721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_457082"
 dense_23/StatefulPartitionedCall┤
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_45746dense_24_45748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_457352"
 dense_24/StatefulPartitionedCallМ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_26_input
░
л
C__inference_dense_24_layer_call_and_return_conditional_losses_46080

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2:::O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
о
л
C__inference_dense_23_layer_call_and_return_conditional_losses_46060

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
И╜2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*0
_input_shapes
:         И╜:::Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
п
Ё
H__inference_sequential_12_layer_call_and_return_conditional_losses_45801

inputs
conv2d_26_45781
conv2d_26_45783
dense_23_45790
dense_23_45792
dense_24_45795
dense_24_45797
identityИв!conv2d_26/StatefulPartitionedCallв dense_23/StatefulPartitionedCallв dense_24/StatefulPartitionedCallв"dropout_11/StatefulPartitionedCallЯ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_26_45781conv2d_26_45783*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_456232#
!conv2d_26/StatefulPartitionedCallП
activation_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_456442
activation_26/PartitionedCallФ
 max_pooling2d_26/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_456032"
 max_pooling2d_26/PartitionedCall■
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_12_layer_call_and_return_conditional_losses_456592
flatten_12/PartitionedCallР
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_456792$
"dropout_11/StatefulPartitionedCall╢
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_23_45790dense_23_45792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_457082"
 dense_23/StatefulPartitionedCall┤
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_45795dense_24_45797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_457352"
 dense_24/StatefulPartitionedCallМ
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ц
╘
H__inference_sequential_12_layer_call_and_return_conditional_losses_45775
conv2d_26_input
conv2d_26_45755
conv2d_26_45757
dense_23_45764
dense_23_45766
dense_24_45769
dense_24_45771
identityИв!conv2d_26/StatefulPartitionedCallв dense_23/StatefulPartitionedCallв dense_24/StatefulPartitionedCallи
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCallconv2d_26_inputconv2d_26_45755conv2d_26_45757*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_456232#
!conv2d_26/StatefulPartitionedCallП
activation_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_456442
activation_26/PartitionedCallФ
 max_pooling2d_26/PartitionedCallPartitionedCall&activation_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_456032"
 max_pooling2d_26/PartitionedCall■
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_12_layer_call_and_return_conditional_losses_456592
flatten_12/PartitionedCall°
dropout_11/PartitionedCallPartitionedCall#flatten_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_11_layer_call_and_return_conditional_losses_456842
dropout_11/PartitionedCallо
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_23_45764dense_23_45766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_23_layer_call_and_return_conditional_losses_457082"
 dense_23/StatefulPartitionedCall┤
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_45769dense_24_45771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_457352"
 dense_24/StatefulPartitionedCallч
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0"^conv2d_26/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_26_input
о
л
C__inference_dense_23_layer_call_and_return_conditional_losses_45708

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
И╜2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*0
_input_shapes
:         И╜:::Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
╨
c
E__inference_dropout_11_layer_call_and_return_conditional_losses_46039

inputs

identity_1\
IdentityIdentityinputs*
T0*)
_output_shapes
:         И╜2

Identityk

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:         И╜2

Identity_1"!

identity_1Identity_1:output:0*(
_input_shapes
:         И╜:Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
Т
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_45679

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constu
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:         И╜2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╢
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:         И╜*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y└
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:         И╜2
dropout/GreaterEqualБ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:         И╜2
dropout/Cast|
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:         И╜2
dropout/Mul_1g
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:         И╜2

Identity"
identityIdentity:output:0*(
_input_shapes
:         И╜:Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
 r
щ
!__inference__traced_restore_46284
file_prefix%
!assignvariableop_conv2d_26_kernel%
!assignvariableop_1_conv2d_26_bias&
"assignvariableop_2_dense_23_kernel$
 assignvariableop_3_dense_23_bias&
"assignvariableop_4_dense_24_kernel$
 assignvariableop_5_dense_24_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1/
+assignvariableop_15_adam_conv2d_26_kernel_m-
)assignvariableop_16_adam_conv2d_26_bias_m.
*assignvariableop_17_adam_dense_23_kernel_m,
(assignvariableop_18_adam_dense_23_bias_m.
*assignvariableop_19_adam_dense_24_kernel_m,
(assignvariableop_20_adam_dense_24_bias_m/
+assignvariableop_21_adam_conv2d_26_kernel_v-
)assignvariableop_22_adam_conv2d_26_bias_v.
*assignvariableop_23_adam_dense_23_kernel_v,
(assignvariableop_24_adam_dense_23_bias_v.
*assignvariableop_25_adam_dense_24_kernel_v,
(assignvariableop_26_adam_dense_24_bias_v
identity_28ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Т
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╞
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╕
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Д
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_26_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ж
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_26_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2з
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_23_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3е
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_23_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4з
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_24_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5е
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_24_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6б
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7г
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8г
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9в
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10о
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11б
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12б
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14г
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15│
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_conv2d_26_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16▒
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_conv2d_26_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17▓
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_23_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18░
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_23_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19▓
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_24_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20░
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_24_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21│
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_26_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22▒
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_26_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_23_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24░
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_23_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_24_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_24_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp░
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27г
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*Б
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╛
I
-__inference_activation_26_layer_call_fn_46011

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_activation_26_layer_call_and_return_conditional_losses_456442
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:X T
0
_output_shapes
:         ╚
 
_user_specified_nameinputs
ї
╛
-__inference_sequential_12_layer_call_fn_45965

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_458012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┌
}
(__inference_dense_24_layer_call_fn_46089

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_457352
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┌
d
H__inference_activation_26_layer_call_and_return_conditional_losses_46006

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╚2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:X T
0
_output_shapes
:         ╚
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_45603

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
d
E__inference_dropout_11_layer_call_and_return_conditional_losses_46034

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constu
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:         И╜2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╢
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:         И╜*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y└
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:         И╜2
dropout/GreaterEqualБ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:         И╜2
dropout/Cast|
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:         И╜2
dropout/Mul_1g
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:         И╜2

Identity"
identityIdentity:output:0*(
_input_shapes
:         И╜:Q M
)
_output_shapes
:         И╜
 
_user_specified_nameinputs
к
F
*__inference_flatten_12_layer_call_fn_46022

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         И╜* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_flatten_12_layer_call_and_return_conditional_losses_456592
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         И╜2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:X T
0
_output_shapes
:         ╚
 
_user_specified_nameinputs
Й%
п
 __inference__wrapped_model_45597
conv2d_26_input:
6sequential_12_conv2d_26_conv2d_readvariableop_resource;
7sequential_12_conv2d_26_biasadd_readvariableop_resource9
5sequential_12_dense_23_matmul_readvariableop_resource:
6sequential_12_dense_23_biasadd_readvariableop_resource9
5sequential_12_dense_24_matmul_readvariableop_resource:
6sequential_12_dense_24_biasadd_readvariableop_resource
identityИ▐
-sequential_12/conv2d_26/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:╚*
dtype02/
-sequential_12/conv2d_26/Conv2D/ReadVariableOpЎ
sequential_12/conv2d_26/Conv2DConv2Dconv2d_26_input5sequential_12/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚*
paddingVALID*
strides
2 
sequential_12/conv2d_26/Conv2D╒
.sequential_12/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype020
.sequential_12/conv2d_26/BiasAdd/ReadVariableOpщ
sequential_12/conv2d_26/BiasAddBiasAdd'sequential_12/conv2d_26/Conv2D:output:06sequential_12/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╚2!
sequential_12/conv2d_26/BiasAdd▒
 sequential_12/activation_26/ReluRelu(sequential_12/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         ╚2"
 sequential_12/activation_26/Relu∙
&sequential_12/max_pooling2d_26/MaxPoolMaxPool.sequential_12/activation_26/Relu:activations:0*0
_output_shapes
:         ╚*
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling2d_26/MaxPoolС
sequential_12/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"    И^  2 
sequential_12/flatten_12/Const▌
 sequential_12/flatten_12/ReshapeReshape/sequential_12/max_pooling2d_26/MaxPool:output:0'sequential_12/flatten_12/Const:output:0*
T0*)
_output_shapes
:         И╜2"
 sequential_12/flatten_12/Reshape▒
!sequential_12/dropout_11/IdentityIdentity)sequential_12/flatten_12/Reshape:output:0*
T0*)
_output_shapes
:         И╜2#
!sequential_12/dropout_11/Identity╘
,sequential_12/dense_23/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_23_matmul_readvariableop_resource* 
_output_shapes
:
И╜2*
dtype02.
,sequential_12/dense_23/MatMul/ReadVariableOp▄
sequential_12/dense_23/MatMulMatMul*sequential_12/dropout_11/Identity:output:04sequential_12/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential_12/dense_23/MatMul╤
-sequential_12/dense_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_23_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02/
-sequential_12/dense_23/BiasAdd/ReadVariableOp▌
sequential_12/dense_23/BiasAddBiasAdd'sequential_12/dense_23/MatMul:product:05sequential_12/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22 
sequential_12/dense_23/BiasAddЭ
sequential_12/dense_23/ReluRelu'sequential_12/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         22
sequential_12/dense_23/Relu╥
,sequential_12/dense_24/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_24_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,sequential_12/dense_24/MatMul/ReadVariableOp█
sequential_12/dense_24/MatMulMatMul)sequential_12/dense_23/Relu:activations:04sequential_12/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_12/dense_24/MatMul╤
-sequential_12/dense_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_12/dense_24/BiasAdd/ReadVariableOp▌
sequential_12/dense_24/BiasAddBiasAdd'sequential_12/dense_24/MatMul:product:05sequential_12/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_12/dense_24/BiasAddж
sequential_12/dense_24/SoftmaxSoftmax'sequential_12/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2 
sequential_12/dense_24/Softmax|
IdentityIdentity(sequential_12/dense_24/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         :::::::` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_26_input"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*├
serving_defaultп
S
conv2d_26_input@
!serving_default_conv2d_26_input:0         <
dense_240
StatefulPartitionedCall:0         tensorflow/serving/predict:х┘
ъ1
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
*t&call_and_return_all_conditional_losses
u__call__
v_default_save_signature"Ў.
_tf_keras_sequential╫.{"class_name": "Sequential", "name": "sequential_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_26_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 1]}, "dtype": "float32", "filters": 200, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_26_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 1]}, "dtype": "float32", "filters": 200, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ў


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"╥	
_tf_keras_layer╕	{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 25, 1]}, "dtype": "float32", "filters": 200, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 25, 1]}}
╫
	variables
trainable_variables
regularization_losses
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_26", "trainable": true, "dtype": "float32", "activation": "relu"}}
Б
	variables
trainable_variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш
	variables
trainable_variables
regularization_losses
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"┘
_tf_keras_layer┐{"class_name": "Flatten", "name": "flatten_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ш
 	variables
!trainable_variables
"regularization_losses
#	keras_api
*&call_and_return_all_conditional_losses
А__call__"╪
_tf_keras_layer╛{"class_name": "Dropout", "name": "dropout_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_11", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
·

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"╙
_tf_keras_layer╣{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24200]}}
Ў

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
┐
0iter

1beta_1

2beta_2
	3decay
4learning_ratemhmi$mj%mk*ml+mmvnvo$vp%vq*vr+vs"
	optimizer
J
0
1
$2
%3
*4
+5"
trackable_list_wrapper
J
0
1
$2
%3
*4
+5"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
5layer_regularization_losses
		variables

trainable_variables
6layer_metrics

7layers
8non_trainable_variables
regularization_losses
9metrics
u__call__
v_default_save_signature
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
-
Еserving_default"
signature_map
+:)╚2conv2d_26/kernel
:╚2conv2d_26/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
:layer_regularization_losses
	variables
trainable_variables
;layer_metrics

<layers
=non_trainable_variables
regularization_losses
>metrics
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
?layer_regularization_losses
	variables
trainable_variables
@layer_metrics

Alayers
Bnon_trainable_variables
regularization_losses
Cmetrics
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Dlayer_regularization_losses
	variables
trainable_variables
Elayer_metrics

Flayers
Gnon_trainable_variables
regularization_losses
Hmetrics
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Ilayer_regularization_losses
	variables
trainable_variables
Jlayer_metrics

Klayers
Lnon_trainable_variables
regularization_losses
Mmetrics
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
о
Nlayer_regularization_losses
 	variables
!trainable_variables
Olayer_metrics

Players
Qnon_trainable_variables
"regularization_losses
Rmetrics
А__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
#:!
И╜22dense_23/kernel
:22dense_23/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
Slayer_regularization_losses
&	variables
'trainable_variables
Tlayer_metrics

Ulayers
Vnon_trainable_variables
(regularization_losses
Wmetrics
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
!:22dense_24/kernel
:2dense_24/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
Xlayer_regularization_losses
,	variables
-trainable_variables
Ylayer_metrics

Zlayers
[non_trainable_variables
.regularization_losses
\metrics
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╗
	_total
	`count
a	variables
b	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ж
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api"┐
_tf_keras_metricд{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
_0
`1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
0:.╚2Adam/conv2d_26/kernel/m
": ╚2Adam/conv2d_26/bias/m
(:&
И╜22Adam/dense_23/kernel/m
 :22Adam/dense_23/bias/m
&:$22Adam/dense_24/kernel/m
 :2Adam/dense_24/bias/m
0:.╚2Adam/conv2d_26/kernel/v
": ╚2Adam/conv2d_26/bias/v
(:&
И╜22Adam/dense_23/kernel/v
 :22Adam/dense_23/bias/v
&:$22Adam/dense_24/kernel/v
 :2Adam/dense_24/bias/v
ю2ы
H__inference_sequential_12_layer_call_and_return_conditional_losses_45919
H__inference_sequential_12_layer_call_and_return_conditional_losses_45948
H__inference_sequential_12_layer_call_and_return_conditional_losses_45775
H__inference_sequential_12_layer_call_and_return_conditional_losses_45752└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
-__inference_sequential_12_layer_call_fn_45982
-__inference_sequential_12_layer_call_fn_45856
-__inference_sequential_12_layer_call_fn_45965
-__inference_sequential_12_layer_call_fn_45816└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
 __inference__wrapped_model_45597╞
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *6в3
1К.
conv2d_26_input         
ю2ы
D__inference_conv2d_26_layer_call_and_return_conditional_losses_45992в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_conv2d_26_layer_call_fn_46001в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_26_layer_call_and_return_conditional_losses_46006в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_26_layer_call_fn_46011в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
│2░
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_45603р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_26_layer_call_fn_45609р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
я2ь
E__inference_flatten_12_layer_call_and_return_conditional_losses_46017в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_flatten_12_layer_call_fn_46022в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╚2┼
E__inference_dropout_11_layer_call_and_return_conditional_losses_46034
E__inference_dropout_11_layer_call_and_return_conditional_losses_46039┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_11_layer_call_fn_46049
*__inference_dropout_11_layer_call_fn_46044┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_23_layer_call_and_return_conditional_losses_46060в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_23_layer_call_fn_46069в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_24_layer_call_and_return_conditional_losses_46080в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_24_layer_call_fn_46089в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:B8
#__inference_signature_wrapper_45883conv2d_26_inputг
 __inference__wrapped_model_45597$%*+@в=
6в3
1К.
conv2d_26_input         
к "3к0
.
dense_24"К
dense_24         ╢
H__inference_activation_26_layer_call_and_return_conditional_losses_46006j8в5
.в+
)К&
inputs         ╚
к ".в+
$К!
0         ╚
Ъ О
-__inference_activation_26_layer_call_fn_46011]8в5
.в+
)К&
inputs         ╚
к "!К         ╚╡
D__inference_conv2d_26_layer_call_and_return_conditional_losses_45992m7в4
-в*
(К%
inputs         
к ".в+
$К!
0         ╚
Ъ Н
)__inference_conv2d_26_layer_call_fn_46001`7в4
-в*
(К%
inputs         
к "!К         ╚е
C__inference_dense_23_layer_call_and_return_conditional_losses_46060^$%1в.
'в$
"К
inputs         И╜
к "%в"
К
0         2
Ъ }
(__inference_dense_23_layer_call_fn_46069Q$%1в.
'в$
"К
inputs         И╜
к "К         2г
C__inference_dense_24_layer_call_and_return_conditional_losses_46080\*+/в,
%в"
 К
inputs         2
к "%в"
К
0         
Ъ {
(__inference_dense_24_layer_call_fn_46089O*+/в,
%в"
 К
inputs         2
к "К         й
E__inference_dropout_11_layer_call_and_return_conditional_losses_46034`5в2
+в(
"К
inputs         И╜
p
к "'в$
К
0         И╜
Ъ й
E__inference_dropout_11_layer_call_and_return_conditional_losses_46039`5в2
+в(
"К
inputs         И╜
p 
к "'в$
К
0         И╜
Ъ Б
*__inference_dropout_11_layer_call_fn_46044S5в2
+в(
"К
inputs         И╜
p
к "К         И╜Б
*__inference_dropout_11_layer_call_fn_46049S5в2
+в(
"К
inputs         И╜
p 
к "К         И╜м
E__inference_flatten_12_layer_call_and_return_conditional_losses_46017c8в5
.в+
)К&
inputs         ╚
к "'в$
К
0         И╜
Ъ Д
*__inference_flatten_12_layer_call_fn_46022V8в5
.в+
)К&
inputs         ╚
к "К         И╜ю
K__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_45603ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_26_layer_call_fn_45609СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ┼
H__inference_sequential_12_layer_call_and_return_conditional_losses_45752y$%*+HвE
>в;
1К.
conv2d_26_input         
p

 
к "%в"
К
0         
Ъ ┼
H__inference_sequential_12_layer_call_and_return_conditional_losses_45775y$%*+HвE
>в;
1К.
conv2d_26_input         
p 

 
к "%в"
К
0         
Ъ ╝
H__inference_sequential_12_layer_call_and_return_conditional_losses_45919p$%*+?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         
Ъ ╝
H__inference_sequential_12_layer_call_and_return_conditional_losses_45948p$%*+?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         
Ъ Э
-__inference_sequential_12_layer_call_fn_45816l$%*+HвE
>в;
1К.
conv2d_26_input         
p

 
к "К         Э
-__inference_sequential_12_layer_call_fn_45856l$%*+HвE
>в;
1К.
conv2d_26_input         
p 

 
к "К         Ф
-__inference_sequential_12_layer_call_fn_45965c$%*+?в<
5в2
(К%
inputs         
p

 
к "К         Ф
-__inference_sequential_12_layer_call_fn_45982c$%*+?в<
5в2
(К%
inputs         
p 

 
к "К         ║
#__inference_signature_wrapper_45883Т$%*+SвP
в 
IкF
D
conv2d_26_input1К.
conv2d_26_input         "3к0
.
dense_24"К
dense_24         