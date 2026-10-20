FROM node:24-alpine AS build
WORKDIR /frontend
COPY apps/frontend/package.json apps/frontend/package-lock.json ./
RUN npm ci
COPY apps/frontend ./
RUN npm run build

FROM node:24-alpine
WORKDIR /frontend
ENV NODE_ENV=production
COPY --from=build /frontend ./
EXPOSE 3000
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0"]
