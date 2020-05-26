# 导入werobot和发送请求的requests
import werobot
import requests

# 设定主要逻辑服务的请求URL
url = "http://39.106.128.55:5000/v1/main_serve/"
# url = "http://161.117.187.37:5000/v1/main_serve/"
# 设定服务超时的时间
TIMEOUT = 3

# 声明微信访问的请求
robot = werobot.WeRoBot(token="doctoraitoken")

# 设置所有请求的入口
@robot.handler
def doctor(message, session):
    try:
        # 获取用户的Id
        uid = message.source
        try:
            # 检查session, 判断用户是否第一次发言
            if session.get(uid, None) != "1":
                # 将添加{uid: "1"}
                session[uid] = "1"
                # 返回用户一个打招呼的话
                return '您好, 我是智能客服小医, 有什么需要帮忙的吗?'
            # 获取message中的用户发言内容
            text = message.content
            # print("1111111111111111",text)
        except:
            # 有时候会出现特殊情况, 用户很可能取消关注后来又再次关注
            # 直接通过session判断, 会发现该用户已经不是第一次发言, 执行message.content语句
            # 真实情况是该用户登录后并没有任何的发言, 获取message.content的时候就会报错
            # 在这种情况下, 我们也通过打招呼的话回复用户
            # print("++++++++++++++++++++++++++++++")
            return '您好, 我是智能客服小医, 有什么需要帮忙的吗?'

        # 向主逻辑服务发送请求, 获得发送的数据体
        data = {"uid": uid, "text": text}
        # print("---------------------",data)
        # 利用requests发送请求
        res = requests.post(url, data=data, timeout=TIMEOUT)
        # 将返回的文本内容返回给用户
        return res.text
    except Exception as e:
        print("出现异常:", e)
        return "对不起, 机器人客服正在休息..."

# 让服务监听在0.0.0.0:80
robot.config["HOST"] = "0.0.0.0"
robot.config["PORT"] = 80
robot.run()

