#encoding=utf-8
import tensorflow as tf
import modeling
import optimization
import os
import time
import datetime
from datetime import timedelta
from bert_config import BertConfig

config=BertConfig()
flags = tf.flags
FLAGS = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES']=config.use_gpu
gpu_options=tf.GPUOptions(allow_growth=True)
## Required parameters
flags.DEFINE_integer(
    "train_data_nums",105000,
    "The num of train datas")
flags.DEFINE_integer(
    "num_labels",4,
    "The num of labels")
flags.DEFINE_integer(
    "num_aspects",20,
    "The num of aspects")
flags.DEFINE_string(
    "data_dir", config.data_dir,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", config.bert_config_file,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name",config.task_name, "The name of the task to train.")

flags.DEFINE_string("vocab_file",config.vocab_file,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir",config.output_dir,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", config.init_checkpoint,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length",config.max_seq_length,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train",config.do_train, "Whether to run training.")

flags.DEFINE_bool("do_eval",config.do_eval, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict",config.do_predict,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size",config.train_batch_size, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size",config.eval_batch_size, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size",config.predict_batch_size, "Total batch size for predict.")

flags.DEFINE_float("learning_rate",config.learning_rate, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2.0,
                   "Total number of training epochs to perform.")
#warmup_proportion：warm up 步数的比例，
# 比如说总共学习100步，warmup_proportion=0.1表示前10步用来warm up，
# warm up时以较低的学习率进行学习(lr = global_step/num_warmup_steps * init_lr)，
# 10步之后以正常(或衰减)的学习率来学习。
#这样做是为了在开始能够找到一个更佳的学习方向
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps",config.save_checkpoints_steps,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
train_examples_size=FLAGS.train_data_nums
num_train_steps=int(int(train_examples_size/FLAGS.train_batch_size)*FLAGS.num_train_epochs)
num_warmup_steps=int(num_train_steps*FLAGS.warmup_proportion)
class BertTextClassify(object):
    def __init__(self,bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,init_checkpoint):
        self.bert_config=bert_config
        self.is_training =is_training
        self.input_ids =input_ids
        self.input_mask =input_mask
        self.segment_ids =segment_ids
        self.labels =labels
        self.num_labels =num_labels
        self.use_one_hot_embeddings =use_one_hot_embeddings
        self.init_checkpoint =init_checkpoint
        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)
        self.one_hot_label=tf.one_hot(self.labels,depth=self.num_labels)
        with tf.name_scope("full_connection"):
            self.all_logits=[]
            self.output_layer=self.model.get_attention_pooled_output()
            self.hidden_size=self.output_layer.shape[-1].value
            if is_training:
                self.output_layer=tf.nn.dropout(self.output_layer,keep_prob=0.9)
            for i in range(FLAGS.num_aspects):
                self.output_weights=tf.get_variable(
                    "output_weights_%d" % i,[self.hidden_size,self.num_labels],
                    initializer=tf.truncated_normal_initializer(stddev=0.02)
                )
                output_bias=tf.get_variable(
                    "output_bias_%d" % i, [self.num_labels],initializer=tf.zeros_initializer()
                )
                logits=tf.matmul(self.output_layer,self.output_weights)
                logits=tf.nn.bias_add(logits,output_bias)
                self.all_logits.append(logits)
            self.pred=tf.concat(self.all_logits,1)
            self.pred=tf.reshape(self.pred,[-1,FLAGS.num_aspects,self.num_labels])
            self.probabilities=tf.nn.softmax(self.pred,axis=-1)
        with tf.name_scope("loss"):
            self.loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.one_hot_label,logits=self.pred)
            self.loss=tf.reduce_sum(self.loss)
        with tf.name_scope("accuracy"):
            self.all_accuracy=[]
            for i in range(FLAGS.num_aspects):
                correct_pred=tf.equal(tf.argmax(self.one_hot_label[:,i],1),tf.argmax(self.probabilities[:,i],1))
                acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32),name="acc_%d" % i)
            self.accuracy=tf.reduce_mean(self.all_accuracy,name='final_acc')
        with tf.name_scope("train_op"):
            # tvars=tf.trainable_variables()
            # grads=tf.gradients(self.loss,tvars)
            # global_step=tf.train.get_or_create_global_step()
            # optimizer=optimization.AdamWeightDecayOptimizer(learning_rate=FLAGS.learning_rate)
            # self.train_op=optimizer.apply_gradients(zip(grads,tvars),global_step)
            self.train_op=optimization.create_optimizer(self.loss,FLAGS.learning_rate,num_train_steps,num_warmup_steps,False)
def _decode_record(record):
    #tf.FixedLenFeature()返回固定长度的Tensor
    name_to_features = {
        "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([FLAGS.num_aspects], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    #将Example协议缓冲区解析为张量
    example=tf.parse_single_example(record,name_to_features)
    #tpu只支持int32
    for name in list(example.keys()):
        t=example[name]
        if t.dtype==tf.int64:
            t=tf.to_int32(t)
        example[name]=t
    return example
def file_based_input_fn_builder(input_file,is_training,batch_size,
                                drop_remainder=False):
    #这部分是对Dataset中的元素进行打乱，组成batch、生成epoch

    #这个函数是用来读TFRecord文件的，dataset中的每一个元素就是一个TFExample
    d=tf.data.TFRecordDataset(input_file)
    #变换成一个map
    d=d.map(lambda x:_decode_record(x))
    if is_training:
        #注意这三句的先后顺序不能改变，否则会造成每个epoch中的数据出现混乱
        #打乱dataset中的数据
        d=d.shuffle(buffer_size=100)
        #将多个元素组成batch_size
        d=d.batch(batch_size,drop_remainder)
        #将整个序列重复多次，将d中的数据重复num_train_epochs次
        d=d.repeat(int(FLAGS.num_train_epochs))
    else:
        d=d.batch(batch_size,drop_remainder)
    return d
def main(_):
    print("Loading data...")
    train_examples_size = FLAGS.train_data_nums
    num_train_steps = int(int(train_examples_size / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
    num_class=FLAGS.num_labels
    its=int(train_examples_size / FLAGS.train_batch_size)
    print('总共需要训练：%d steps' % num_train_steps)

    input_path_train="%s/train.tf_record" % FLAGS.output_dir
    input_path_eval="%s/eval.tf_record" % FLAGS.output_dir
    input_path_predict="%s/eval.tf_record" % FLAGS.output_dir

    #加载tf_record，获得训练数据的dataset
    iterator=file_based_input_fn_builder(input_path_train,drop_remainder=True,batch_size=FLAGS.train_batch_size,is_training=True)
    #从dataset中实例化一个迭代器，这个迭代器只能从头到尾读取一次
    #之所以这里和验证集使用的迭代器是不同的迭代器是因为训练集只需迭代一遍
    #而验证集和测试集需迭代多次
    iterator=iterator.make_one_shot_iterator()
    #从迭代器中取出一个值，由于使用的是非Eager模式，所以每个features并不是一个值
    #需要通过sess.run()来获取值
    features=iterator.get_next()
    bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    is_training=FLAGS.do_train
    init_checkpoint=FLAGS.init_checkpoint
    use_one_hot_embeddings=False

    input_ids=tf.placeholder(tf.int32,[None,FLAGS.max_seq_length],name='input_ids')
    input_mask_=tf.placeholder(tf.int32,[None,FLAGS.max_seq_length],name='input_mask')
    label_ids=tf.placeholder(tf.int32,[None,FLAGS.num_aspects],name='label_ids')

    model=BertTextClassify(bert_config,is_training,input_ids,input_mask_,
                           None,label_ids,num_class,
                           use_one_hot_embeddings,init_checkpoint)

    #加载tf_record，获得验证数据的dataset
    #这里有个缺陷是每次只能以batch_size大小的数据集作为验证集
    iterator_eval=file_based_input_fn_builder(input_path_eval,drop_remainder=False,
                                         batch_size=FLAGS.eval_batch_size,is_training=False)
    #通过下面这种方法生成的迭代器在使用前必须先通过sess.run()来初始化
    #它的好处是可以将placeholder带入Iterator中，这样可以方便我们通过参数快速定义新的Iterator
    iterator_eval=iterator_eval.make_initializable_iterator()
    features_eval=iterator_eval.get_next()
    init_op_eval=iterator_eval.initializer

    # 加载tf_record，获得测试数据的dataset
    # 这里有个缺陷是每次只能以batch_size大小的数据集作为验证集
    iterator_predict = file_based_input_fn_builder(input_path_predict, drop_remainder=False,
                                                batch_size=FLAGS.eval_batch_size, is_training=False)
    # 通过下面这种方法生成的迭代器在使用前必须先通过sess.run()来初始化
    # 它的好处是可以将placeholder带入Iterator中，这样可以方便我们通过参数快速定义新的Iterator
    iterator_predict = iterator_predict.make_initializable_iterator()
    features_predict = iterator_predict.get_next()
    init_op_predict = iterator_predict.initializer

    #这部分是加载Bert的预训练模型并创建session
    tvars=tf.trainable_variables()
    if init_checkpoint:
        (assignment_map,initialized_variable_names)=modeling.get_assignment_map_from_checkpoint(
            tvars,init_checkpoint
        )
        tf.train.init_from_checkpoint(init_checkpoint,assignment_map)
    session=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    #打印网络结构中的变量
    print("****Trainable Variable****")
    for var in tvars:
        if var.name in initialized_variable_names:
            init_string="，*INIT_FROM_CKPT*"
            print("name ={0}，shape={1}{2}".format(var.name,var.shape,init_string))


    print("bert classifier model will start train ........")
    #设置最多保存ckpt的个数，默认是5
    saver=tf.train.Saver(max_to_keep=5)
    #若存在检测点则在检测点的基础上继续训练
    #获得所有检测点名称
    ckpt=tf.train.get_checkpoint_state(config.output_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session,ckpt.model_checkpoint_path)

    start_time=time.time()
    i=0
    for index in range(int(FLAGS.num_train_epochs)):
        all_its=its*(index+1)
        while i<all_its:
            feed_dict={
                model.input_ids:features["input_ids"],
                model.input_mask:features["input_mask"],
                model.labels:features["label_ids"]
            }
            feed_dict=session.run(feed_dict)
            _,loss_train,acc_train=session.run([model.train_op,model.loss,model.accuracy]
                                               ,feed_dict=feed_dict)
            i=i+1
            if i%1000==0:
                end_time=time.time()
                time_dif=end_time-start_time
                time_dif=timedelta(seconds=int(round(time_dif)))
                msg='Iter: {0:>6}, Train Loss: {1:>6.2}，，Train Acc：{2:>6.2}，'+'  Cost：{3}   Time:{4}'
                print(msg.format(i,loss_train,acc_train,time_dif,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                start_time=time.time()
            if (i%1000==0 and i>0) or i==num_train_steps:
                save_location=os.path.join(FLAGS.output_dir,'model.ckpt')
                saver.save(session,save_location,global_step=i)
            if FLAGS.do_eval and (i%5000==0 or i==num_train_steps):
                session.run(init_op_eval)
                feed_dict = {
                    model.input_ids: features_eval["input_ids"],
                    model.input_mask: features_eval["input_mask"],
                    model.labels: features_eval["label_ids"]
                }
                feed_dict=session.run(feed_dict)
                loss_eval,acc_eval=session.run([model.loss,model.accuracy],feed_dict=feed_dict)
                print("eval loss is：%f...eval acc is：%f" % (loss_eval,acc_eval))
    session.close()
if __name__ == '__main__':
    tf.app.run()