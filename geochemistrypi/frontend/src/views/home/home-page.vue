<template>
	<div class="home-page">
		<div class="text-wrapper">
			<div>
				<h1><img class="logo" src="@/assets/imgs/onlyLogo.png" />Geochemistry π</h1>
				<p>A Python Framework For Geochemistry</p>
			</div>

		</div>
	</div>
</template>
<style lang="scss" scoped>
.home-page {
	height: calc(100vh - 75px);
	width: 100%;
	background: url('@/assets/imgs/bg.jpg');
	background-size: cover;
	overflow: hidden;

	.text-wrapper {
		margin-left: 115px;
		margin-top: 180px;
		font-size: 30px;
		color: #fff;
		display: flex;

		.logo {
			width: 96px;
			line-height: 144px;
			margin-right: 20px
		}
		p{
			margin-left: 116px;
		}
		h1 {
			color: #F25B28;
			font-size: 90px;
			font-weight: 500;
		}
	}
}
</style>