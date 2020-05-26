# 服务框架使用Flask
# 导入相关的包
from flask import Flask
from flask import request
app = Flask(__name__)

# 导入发送http请求的requests工具
import requests

# 导入redis
import redis

# 导入json工具
import json

# 导入已经编写好的Unit API文件
from unit import unit_chat

# 导入操作neo4j数据库的工具
from neo4j import GraphDatabase

# 从配置文件config.py导入需要的若干配置信息
# 导入neo4j的相关信息
from config import NEO4J_CONFIG
# 导入redis的相关信息
from config import REDIS_CONFIG
# 导入句子相关模型服务的请求地址
from config import model_serve_url
# 导入句子相关模型服务的超时时间
from config import TIMEOUT
# 导入规则对话模型的加载路径
from config import reply_path
# 导入用户对话信息保存的过期时间
from config import ex_time

# 建立redis的连接池
pool = redis.ConnectionPool( **REDIS_CONFIG)

# 初始化neo4j的驱动对象
_driver = GraphDatabase.driver( **NEO4J_CONFIG)


# 查询neo4j图数据的函数
def query_neo4j(text):
    ''''
    功能: 根据用户对话文本中可能存在的疾病症状, 来查询图数据库, 返回对应的疾病名称
    text: 用户输入的文本语句
    return: 用户描述的症状所对应的的疾病名称列表
    '''
    # 开启一个会话session来操作图数据库
    with _driver.session() as session:
        # 构建查询的cypher语句, 匹配句子中存在的所有症状节点
        # 保存这些临时的节点, 并通过关系dis_to_sym进行对应疾病名称的查找, 返回找到的疾病名称列表
        cypher = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH \
                 a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 5" %text
        # 通过会话session来运行cypher语句
        record = session.run(cypher)
        # 从record中读取真正的疾病名称信息, 并封装成List返回
        result = list(map(lambda x: x[0], record))
    return result


# 主要逻辑服务类Handler类
class Handler(object):
    def __init__(self, uid, text, r, reply):
        '''
        uid: 用户唯一标识uid
        text: 标识该用户本次输入的文本信息
        r: 代表redis数据库的一个链接对象
        reply: 规则对话模板加载到内存中的对象(字典对象)
        '''
        self.uid = uid
        self.text = text
        self.r = r
        self.reply = reply

    # 编写非首句处理函数, 该用户不是第一句问话
    def non_first_sentence(self, previous):
        '''
        previous: 代表该用户当前语句的上一句文本信息
        '''
        # 尝试请求语句模型服务, 如果失败, 打印错误信息
        # 在此处打印信息, 说明服务已经可以进入到首句处理函数中
        print("准备请求句子相关模型服务!")
        try:
            data = {"text1": previous, "text2": self.text}
            # 直接向语句服务模型发送请求
            result = requests.post(model_serve_url, data=data, timeout=TIMEOUT)
            # 如果回复为空, 说明服务暂时不提供信息, 转去百度机器人回复
            if not result.text:
                return unit_chat(self.text)
            # 此处打印信息, 说明句子相关模型服务请求成功且不为空
            print("句子相关模型服务请求成功, 返回结果为:", result.text)
        except Exception as e:
            print("模型服务异常:", e)
            return unit_chat(self.text)

        # 此处打印信息, 说明程序已经准备进行neo4j数据库查询
        print("骑牛模型服务后, 准备请求neo4j查询服务!")        
        # 查询图数据库, 并得到疾病名称的列表结果
        s = query_neo4j(self.text)
        # 此处打印信息, 说明已经成功获得了neo4j的查询结果
        print("neo4j查询服务请求成功, 返回结果是:", s)
        # 判断如果结果为空, 继续用百度机器人回复
        if not s:
            return unit_chat(self.text)
        # 如果结果不是空, 从redis中获取上一次已经回复给用户的疾病名称
        old_disease = self.r.hget(str(self.uid), "previous_d")
        # 如果曾经回复过用户若干疾病名称, 将新查询的疾病和已经回复的疾病做并集, 再次存储
        # 新查询的疾病, 要和曾经回复过的疾病做差集, 这个差集再次回复给用户
        if old_disease:
            # new_disease是本次需要存储进redis数据库的疾病, 做并集得来
            new_disease = list(set(s) | set(eval(old_disease)))
            # 返回给用户的疾病res, 是本次查询结果和曾经的回复结果之间的差集
            res = list(set(s) - set(eval(old_disease)))
        else:
            # 如果曾经没有给该用户的回复疾病, 则存储的数据和返回给用户的数据相同, 都是从neo4j数据库查询返回的结果
            res = new_disease = list(set(s))

        # 将new_disease存储进redis数据库中, 同时覆盖掉之前的old_disease
        self.r.hset(str(self.uid), "previous_d", str(new_disease))
        # 设置redis数据的过期时间
        self.r.expire(str(self.uid), ex_time)

        # 此处打印信息, 说明neo4j查询后已经处理完了redis任务, 开始使用规则对话模板
        print("使用规则对话模板进行返回对话的生成!")
        # 将列表转化为字符串, 添加进规则对话模板中返回给用户
        if not res:
            return self.reply["4"]
        else:
            res = ",".join(res)
            return self.reply["2"] %res

    # 编码首句请求的代码函数
    def first_sentence(self):
        # 此处打印信息, 说明程序逻辑进入了首句处理函数, 并且马上要进行neo4j查询
        print("该用户近期首次发言, 不必请求模型服务, 准备请求neo4j查询服务!")
        # 直接查询neo4j图数据库, 并得到疾病名称列表的结果
        s = query_neo4j(self.text)
        # 此处打印信息, 说明已经成功完成了neo4j查询服务
        print("neo4j查询服务请求成功, 返回结果:", s)
        # 判断如果结果为空列表, 再次访问百度机器人
        if not s:
            return unit_chat(self.text)

        # 将查询回来的结果存储进redis, 并且做为下一次访问的"上一条语句"previous
        self.r.hset(str(self.uid), "previous_d", str(s))
        # 设置数据库的过期时间
        self.r.expire(str(self.uid), ex_time)
        # 将列表转换为字符串, 添加进规则对话模板中返回给用户
        res = ",".join(s)
        # 此处打印信息, 说明neo4j查询后有结果并且非空, 接下来将使用规则模板进行对话生成
        print("使用规则对话生成模板进行返回对话的生成!")
        return self.reply["2"] %res


# 定义主要逻辑服务的主函数
# 首先设定主要逻辑服务的路由和请求方法
@app.route('/v1/main_serve/', methods=["POST"])
def main_serve():
    # 此处打印信息, 说明werobot服务成功的发送了请求
    print("已经进入主要逻辑服务, werobot服务正常运行!")
    # 第一步接收来自werobot服务的相关字段, uid: 用户唯一标识, text: 用户输入的文本信息
    uid = request.form['uid']
    text = request.form['text']

    # 从redis连接池中获得一个活跃的连接
    r = redis.StrictRedis(connection_pool=pool)

    # 获取该用户上一次说的话(注意: 可能为空)
    previous = r.hget(str(uid), "previous")
    # 将当前输入的text存入redis, 作为下一次访问时候的"上一句话"
    r.hset(str(uid), "previous", text)

    # 此处打印信息, 说明redis能够正常读取数据和写入数据
    print("已经完成了初次会话管理, redis运行正常!")
    # 将规则对话模板的文件Load进内存
    reply = json.load(open(reply_path, "r"))

    # 实例化Handler类
    handler = Handler(uid, text, r, reply)

    # 如果上一句话存在, 调用非首句服务函数
    if previous:
        return handler.non_first_sentence(previous)
    # 如果上一句话不存在, 调用首句服务函数
    else:
        return handler.first_sentence()






# if __name__ == '__main__':
#     text = "我最近腹痛!"
#     result = query_neo4j(text)
#     print("疾病列表:", result)


