<template>
  <div class="guide-page">
    <div class="text-wrapper">
      <h1>Meet Our Team Member</h1>
      <p>Data Organization: ZJU Earth Data Team</p>
    </div>
    <div class="member-group">
      <el-card body-class="member-box" v-for="item in memberInfo" :key="item.id">
        <div class="member-img"><img :src="item.img" /></div>
        <div class="member-name">{{ item.name }}</div>
        <div class="member-responsibility">{{ item.responsibility }}</div>
        <div class="member-from">{{ item.from }}</div>
      </el-card>
    </div>
  </div>
</template>
<script setup>
import { memberInfo } from './config'
import { ElCard, ElSpace } from 'element-plus'
</script>
<style lang="scss" scoped>
.guide-page {
  .text-wrapper {
    margin-top: 20px;
    text-align: center;
  }
  .member-group {
    margin: 40px 32px;
    display: flex;
    flex-wrap: wrap;
  }
  :deep() {
    .el-card {
      width: calc((100% - 120px) / 4);
      box-sizing: border-box;
      margin-right: 40px;
      margin-bottom: 40px;
      &:nth-of-type(4n) {
        margin-right: 0px;
      }
    }
    .member-box {
      .member-img {
        img {
          height: 200px;
          object-fit: cover;
          object-position: center;
        }
      }
      .member-name {
        font-size: 18px;
        font-weight: 500;
      }
    }
  }
}
</style>
