<template>
  <div class="team-merber">
    <!-- paragraph -->
    <div class="paragraph">
      <p class="paragraph-title">Meet Our Team Member</p>
      <p class="paragraph-desc">With these team members working together, we were able to complete this awesome
        project!</p>
    </div>
    <!-- head of team -->
    <div class="head-of-team">
      <div class="photo">
        <img src="@/assets/imgs/hecan.png" />
      </div>
      <div>
        <p class="add-circle name">Can He</p>
        <p class="other-info">Head of Whole Team</p>
        <p class="school">National University of Singapore </p>
      </div>
    </div>
    <!-- Technical Group -->
    <GroupMember name="Technical Group" :list="TechnicalGroup" />
    <!-- Product Group -->
    <GroupMember name="Product Group" :list="ProductGroup" />
    <!-- Adviser -->
    <div class="adviser">
      <div class="title">
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M40 42V38C40 35.8783 39.1571 33.8434 37.6569 32.3431C36.1566 30.8429 34.1217 30 32 30H16C13.8783 30 11.8434 30.8429 10.3431 32.3431C8.84285 33.8434 8 35.8783 8 38V42M32 14C32 18.4183 28.4183 22 24 22C19.5817 22 16 18.4183 16 14C16 9.58172 19.5817 6 24 6C28.4183 6 32 9.58172 32 14Z"
            stroke="#F25B28" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        </svg>

        <span class="text">Adviser</span>
      </div>
      <div class="intro-wrapper">
        <div class="photo"><img src="@/assets/imgs/zhangzhou.png" /></div>
        <div class="info">
          <p class="personal">
            <span class="add-circle english-name">ZhangZhou</span>
            <span class="chinese-name">张舟</span>
            <span class="university">Zhejiang University</span>
          </p>
          <p class="other"> 百人计划研究员，博士生导师，浙江大学地球科学学院</p>
        </div>
      </div>
    </div>
    <!-- History Contributors -->
    <div class="history-contributors">
      <div class="title">
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M34 42V38C34 35.8783 33.1571 33.8434 31.6569 32.3431C30.1566 30.8429 28.1217 30 26 30H10C7.87827 30 5.84344 30.8429 4.34315 32.3431C2.84285 33.8434 2 35.8783 2 38V42M46 42V38C45.9987 36.2275 45.4087 34.5055 44.3227 33.1046C43.2368 31.7037 41.7163 30.7031 40 30.26M32 6.26C33.7208 6.7006 35.2461 7.7014 36.3353 9.10462C37.4245 10.5078 38.0157 12.2337 38.0157 14.01C38.0157 15.7863 37.4245 17.5122 36.3353 18.9154C35.2461 20.3186 33.7208 21.3194 32 21.76M26 14C26 18.4183 22.4183 22 18 22C13.5817 22 10 18.4183 10 14C10 9.58172 13.5817 6 18 6C22.4183 6 26 9.58172 26 14Z"
            stroke="#F25B28" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <span class="text">History Contributors</span>
      </div>
      <div class="list-wrapper">
        <div class="item" v-for="item in HistoryContributors" :key="item.id">
          <div class="item-name  add-circle">{{ item.name }}</div>
          <div class="item-from">{{ item.from }}</div>
        </div>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import GroupMember from './group-member.vue'
import { TechnicalGroup, ProductGroup, HistoryContributors } from '../config'
</script>
<style lang="scss" scoped>
.team-merber {
  .paragraph {
    margin-bottom: 50px;

    &-title {
      font-size: 48px;
      font-weight: 600;
      text-align: center;
      line-height: 54px;
      margin-bottom: 10px;
    }

    &-desc {
      color: #AEB4AF;
      font-size: 16px;
      line-height: 26px;
      text-align: center;


    }
  }

  .head-of-team {
    display: flex;
    justify-content: center;
    align-items: center;

    .photo {
      img {
        width: 166px;
        margin-right: 36px;
      }
    }

    .name {
      display: flex;
      align-items: center;
      font-family: Jost;
      font-size: 24px;
      font-weight: 600;
      line-height: 38px;
      margin-bottom: 8px;
    }

    .other-info {
      font-size: 24px;
      font-weight: 500;
      line-height: 30px;
      color: #F25B28;
    }

    .school {
      font-size: 24px;
      font-weight: 400;
      line-height: 38px;

    }
  }

  .adviser {
    margin-bottom: 60px;

    .title {
      display: flex;
      align-items: center;
      margin-bottom: 20px;

      .text {
        font-size: 36px;
        font-weight: 500;
        line-height: 52px;
        text-align: left;
        color: #0E140F;
        margin-left: 40px;
      }
    }

    .intro-wrapper {
      border-radius: 80px;
      display: flex;
      align-items: center;
      background: #FFEDE761;
      padding: 54px 190px;

      .photo {
        width: 160px;
        border-radius: 40px;
        overflow: hidden;
        margin-right: 120px;

        img {
          width: 100%;
        }
      }

      .personal {
        display: flex;
        align-items: center;
        margin-bottom: 8px;

        .english-name {
          font-size: 20px;
          font-weight: 600;
          line-height: 38px;
          margin-right: 16px
        }

        .chinese-name {
          font-size: 20px;
          font-weight: 600;
          line-height: 38px;
          margin-right: 16px
        }

        .university {
          font-size: 20px;
          font-weight: 400;
          line-height: 38px;

        }
      }

      .other {
        font-size: 20px;
        font-weight: 400;
        line-height: 38px;

      }
    }
  }

  .history-contributors {
    .title {
      display: flex;
      align-items: center;
      margin-bottom: 20px;

      .text {
        font-size: 36px;
        font-weight: 500;
        line-height: 52px;
        text-align: left;
        color: #0E140F;
        margin-left: 40px;
      }
    }

    .list-wrapper {
      width: 50%;
      border-radius: 80px;
      background: #FDC3AD57;
      min-height: 800px;
      max-width: 650px;
      padding: 58px 100px;

      .item {
        margin-bottom: 24px;

        &-name {
          display: flex;
          align-items: center;
          font-family: Jost;
          font-size: 20px;
          font-weight: 600;
          line-height: 38px;

        }

        &-from {
          font-size: 20px;
          font-weight: 400;
          line-height: 38px;

        }
      }
    }
  }

  .add-circle:before {
    display: inline-block;
    height: 20px;
    content: "";
    width: 20px;
    border: 2px solid #F25B28;
    border-radius: 10px;
    margin-right: 16px;
  }
}
</style>