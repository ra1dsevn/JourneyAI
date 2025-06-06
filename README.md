JourneyAI
项目概述
JourneyAI 是一个基于人工智能的旅行规划工具，通过先进的语言模型为用户提供个性化的旅行推荐和行程规划，针对用户偏好（如预算、旅行风格、兴趣）生成高质量建议。

功能特性

个性化推荐：根据用户输入（如预算、兴趣、旅行目的地）生成定制化行程。
高效交互：提供精准、快速的旅行建议。
多平台支持：支持 USA 和加拿大旅行政策查询，计划扩展至更多地区。

技术栈

核心框架：LangChain
编程语言：Python
前端：React（Node.js）
依赖管理：pip（requirements.txt）、npm

安装指南

克隆仓库：git clone https://github.com/ra1dsevn/JourneyAI.git
cd JourneyAI


安装后端依赖：pip install -r requirements.txt


安装前端依赖：cd itinera-frontend
npm install
cd ..



使用方法

启动后端服务：python main.py


启动前端服务：cd itinera-frontend
npm start


访问前端：
打开浏览器，进入 http://localhost:3000。


输入旅行相关查询（如“为预算 $1000 的家庭设计 3 天纽约行程”）。
输出示例：纽约 3 天行程：
第 1 天：参观自由女神像、中央公园...
第 2 天：探索大都会博物馆、时代广场...
第 3 天：布鲁克林大桥、当地美食体验...



贡献指南
欢迎贡献！请遵循以下步骤：

Fork 仓库。
创建特性分支（git checkout -b feature/xxx）。
提交更改（git commit -m "添加 xxx 功能"）。
推送至分支（git push origin feature/xxx）。
创建 Pull Request。

注意事项

确保 itinera-frontend 目录包含 package.json。
需 GPU 环境支持部分功能，推荐使用云端服务。

联系方式

作者：ra1dsevn
邮箱：ra1dsevn@gmail.com
问题反馈：请在 GitHub Issues 提交。

许可
MIT License
