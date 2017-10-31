

docker build -t human_pose:%1 . -f ./Dockerfile

docker tag human_pose:%1 uhub.service.ucloud.cn/bj601/human_pose:%1

docker push uhub.service.ucloud.cn/bj601/human_pose:%1
