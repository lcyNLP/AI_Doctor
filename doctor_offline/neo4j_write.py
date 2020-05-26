import os
import fileinput
from neo4j import GraphDatabase
from config import NEO4J_CONFIG

driver = GraphDatabase.driver( **NEO4J_CONFIG)

# 导入数据的函数
def _load_data(path):
    """
    功能：将path参数目录下的csv文件以指定的格式加载到内存中
    path: 经历了命名实体审核后，所有的疾病-症状的csv文件
    return: 返回疾病:症状的字典 {疾病1:[症状1,症状2,...],疾病2:[症状1,症状2,...]}
    """

    # 获得所有疾病对应的csv文件的列表
    disease_csv_list = os.listdir(path)

    # 将文件名的后缀.csv去除掉，获得所有疾病名称的列表
    disease_list = list(map(lambda x: x.split('.')[0], disease_csv_list))

    # 将每一种疾病对应的所有症状放在症状列表中
    symptom_list = []
    for disease_csv in disease_csv_list:
        # 将一个疾病文件中所有的症状提取到一个列表中
        symptom = list(map(lambda x: x.strip(), fileinput.FileInput(os.path.join(path, disease_csv))))

        # 过滤掉所有长度异常的症状名称
        symptom = list(filter(lambda x: 0<len(x)<100, symptom))
        symptom_list.append(symptom)

    return dict(zip(disease_list, symptom_list))


# 写入图数据库的函数
def write(path):
    """
    功能: 将csv数据全部写入neo4j图数据库中
    path: 经历了命名实体审核后，所有的疾病-症状的csv文件
    """

    # 导入数据成为字典类型
    disease_symptom_dict = _load_data(path)

    # 开启一个会话，进行数据库的操作
    with driver.session() as session:
        for key, value in disease_symptom_dict.items():
            # 创建疾病名的节点
            cypher = "MERGE (a:Disease{name:%r}) RETURN a" %key
            session.run(cypher)
            # 循环处理症状名称的列表
            for v in value:
                # 创建症状的节点
                cypher = "MERGE (b:Symptom{name:%r}) RETURN b" %v
                session.run(cypher)
                # 创建疾病名-疾病症状之间的关系
                cypher = "MATCH (a:Disease{name:%r}) MATCH (b:Symptom{name:%r}) \
			 WITH a,b MERGE (a)-[r:dis_to_sym]-(b)" %(key, v)
                session.run(cypher)

        # 创建Disease节点的索引
        cypher = "CREATE INDEX ON:Disease(name)"
        session.run(cypher)
        # 创建Symptom节点的索引
        cypher = "CREATE INDEX ON:Symptom(name)"
        session.run(cypher)


if __name__ == '__main__':
    path = "./structured/reviewed/"
    write(path)

























