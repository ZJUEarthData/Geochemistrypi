<template>
  <div class="guide-page">
    <div class="text-wrapper">
      <h1>About Us</h1>
      <p>Home > About Us</p>
    </div>
    <div class="content-wrapper">
      <!-- paragraph -->
      <div class="paragraph">
        <p class="paragraph-title">About Project</p>
        <p class="paragraph-desc">Our program is efficient and convenient, welcome to use our program!</p>
      </div>
      <!-- what-it-is -->
      <div class="what-it-is">
        <div class="left">
          <p class="title">What it is</p>
          <p class="desc">Geochemistry π is an open-sourced highly automated machine learning Python
            framework dedicating to build up MLOps level 1 software product for data-driven geochemistry discovery on
            tabular data.</p>
        </div>
        <div class="right">
          <div class="small-fanshaped" />
          <div class="big-fanshaped" />
        </div>
      </div>
      <!-- Core-capabilities -->
      <div class="core-capabilities">
        <div class="left">
          <div class="small-graph" />
          <div class="big-graph" />
        </div>
        <div class="right">
          <div class="title">Core capabilities</div>
          <div class="list">
            <div class="item">
              <CircleCheck :style="{ color: '#F25B28', marginRight: 8, width: 20 }" />
              Continous Training
            </div>
            <div class="item">
              <CircleCheck :style="{ color: '#F25B28', marginRight: 8, width: 20 }" />Machine Learning Lifecycle
              Management
            </div>
            <div class="item">
              <CircleCheck :style="{ color: '#F25B28', marginRight: 8, width: 20 }" />Model Inference
            </div>
          </div>
        </div>
      </div>
      <!-- key-features -->
      <div class="key-features">
        <div class="left">
          <div class="title">Key features</div>
          <div class="list">
            <div class="item">
              <div class="item-title">
                <DArrowRight :style="{ color: '#F25B28', marginRight: 42, width: 25 }" />
                Easy to use
              </div>
              <div class="item-desc">The automation of data mining process provides the users with simple number options
                to choose.</div>
            </div>
            <div class="item">
              <div class="item-title">
                <DArrowRight :style="{ color: '#F25B28', marginRight: 42, width: 25 }" />Machine Learning Lifecycle
                Extensible
              </div>
              <div class="item-desc">It allows appending new algorithms through Scikit-learn with automatic hyper
                parameter searching by FLAML and Ray.</div>
            </div>
            <div class="item">
              <div class="item-title">
                <DArrowRight :style="{ color: '#F25B28', marginRight: 42, width: 25 }" />Traceable
              </div>
              <div class="item-desc">It integrates MLflow to build special storage mechanism to streamline the
                end-to-end
                machine learning lifecycle.</div>
            </div>
          </div>
        </div>
        <div class="right">
          <div class="top-graph" />
          <div class="bottom-graph" />
        </div>
      </div>
      <!-- team-member -->
      <TeamMember />

    </div>
  </div>
</template>
<script lang="ts" setup>
import TeamMember from './component/team-member.vue'
</script>
<style lang="scss" scoped>
.guide-page {
  .text-wrapper {
    color: white;
    background: black;
    height: 100vh;
    width: 100%;
    background: url('../../assets/imgs/about_bg.jpg') no-repeat;
    background-size: cover;
    overflow: hidden;

    h1 {
      font-weight: 500;
      font-size: 58px;
      line-height: 84px;
      margin-top: calc(40% - 112px);
      margin-left: 100px;
    }

    p {
      font-weight: 500;
      font-size: 16px;
      line-height: 30px;
      margin-left: 100px;
    }
  }

  .content-wrapper {
    padding: 60px 100px;

    .paragraph {
      margin-bottom: 20px;

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

    .what-it-is {
      padding: 20px 0;
      margin-bottom: 90px;
      display: flex;

      .left {
        padding-left: 35px;
        display: flex;
        flex-direction: column;
        flex: 1;

        .title {
          font-size: 36px;
          font-weight: 500;
          line-height: 52px;
          margin-bottom: 10px;
        }

        .desc {
          font-size: 20px;
          font-weight: 400;
          line-height: 38px;

        }
      }

      .right {
        margin-left: 126px;
        width: 260px;
        height: 260px;
        position: relative;

        .small-fanshaped {
          position: absolute;
          bottom: -190px;
          left: -190px;
          width: 380px;
          height: 380px;
          border-radius: 0 50% 50% 0;
          background: #FD9875;
          clip-path: inset(0 0 50% 50%);
        }

        .big-fanshaped {
          position: absolute;
          bottom: -260px;
          left: -260px;
          width: 520px;
          height: 520px;
          border-radius: 0 50% 50% 0;
          background: #FFA482EB;
          clip-path: inset(0 0 50% 50%);
        }
      }
    }

    .core-capabilities {
      display: flex;
      margin-bottom: 90px;
      justify-content: space-between;

      .left {
        height: 292px;
        width: 360px;
        position: relative;

        .big-graph {
          width: 360px;
          height: 292px;
          border-bottom-left-radius: 180px;
          border-top-right-radius: 180px;
          background-color: #fde6df;
        }

        .small-graph {
          position: absolute;
          left: 40px;
          bottom: 0;
          z-index: 1;
          width: 280px;
          height: 292px;
          border-bottom-left-radius: 180px;
          border-top-right-radius: 180px;
          background-color: #fbd1c4;
        }
      }

      .right {
        padding-top: 60px;
        margin-right: 10%;

        .title {
          font-size: 36px;
          font-weight: 500;
          line-height: 52px;
          margin-bottom: 20px;
        }

        .list {
          border-left: 6px #F25B28 solid;
          padding: 8px 26px;

          .item {
            font-size: 20px;
            line-height: 26px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;

            &:last-child {
              margin-bottom: 0px;
            }
          }
        }
      }
    }

    .key-features {
      display: flex;
      margin-bottom: 130px;

      .left {
        flex: 1;

        .title {
          padding-left: 44px;
          font-size: 36px;
          font-weight: 500;
          line-height: 52px;
          color: #0E140F;
          margin-bottom: 45px;
        }

        .item {
          margin-bottom: 30px;

          &-title {
            display: flex;
            align-items: center;
            font-size: 20px;
            font-weight: 600;
            line-height: 24px;
            margin-bottom: 10px;
            color: #000;
          }

          &-desc {
            font-size: 20px;
            font-weight: 400;
            line-height: 24px;
            text-align: left;
            padding-left: 62px;
            color: #000;
          }
        }
      }

      .right {
        margin-left: 40px;

        .top-graph {
          width: 240px;
          height: 185px;
          border-radius: 120px 100px 0px 100px;
          background: #F4DED6;
          margin-bottom: 20px;
        }

        .bottom-graph {
          width: 260px;
          height: 240px;
          border-radius: 1px 86px 1px 62px;
          background: #FD9875;

        }
      }
    }
  }
}
</style>
