# Frontend (React App)
FROM node:23 AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install --force

COPY . .

ENV CI=false
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 3001

CMD ["nginx", "-g", "daemon off;"]
