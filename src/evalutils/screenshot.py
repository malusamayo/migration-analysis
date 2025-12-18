import asyncio
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

async def take_screenshot(url, output_path):
	"""Take screenshots of a webpage using BrowserSession."""
	# Create a browser session
	session = BrowserSession(
		browser_profile=BrowserProfile(
			headless=True,
		)
	)

	# Start the browser
	await session.start()

	try:
		await session.navigate_to(url)
		print(f'Navigated to {url}')
		await session.take_screenshot(
			path=output_path,
			full_page=True,
		)

	finally:
		# Clean up
		await session.kill()


if __name__ == '__main__':
	asyncio.run(take_screenshot("https://www.google.com", "google_screenshot.png"))